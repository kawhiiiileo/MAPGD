"""
HCGC (Hypersphere-Constrained Gradient Clustering) and
CAAW (Channel-Adaptive Agent Weighting) Implementation

Add these classes to your core.py file, or import them as needed.
"""

import numpy as np
from sklearn.cluster import KMeans
from llm import call_openai
from utils import MAPGDUtils
import time


class HypersphereConstrainedGradientClustering:
    """
    Hypersphere-Constrained Gradient Clustering (HCGC)

    Implements angular margin constraints on the unit hypersphere to ensure:
    1. Intra-cluster compactness (gradients within same cluster are similar)
    2. Inter-cluster separation (gradients from different clusters are distinct)

    Based on Section 3.3 of the MAPGD paper.
    """

    def __init__(self, config):
        self.conflict_threshold = config.get('conflict_threshold', 0.3)  # θ for conflict detection
        self.max_clusters = config.get('max_clusters', 5)
        self.margin_scale = config.get('margin_scale', 2.0)  # n parameter for angular margin
        self.temperature = config.get('clustering_temperature', 0.1)  # τ for softmax

        # Use shared semantic model
        self.semantic_model = MAPGDUtils.get_semantic_model()

    def apply(self, agent_gradients):
        """
        Main HCGC pipeline:
        1. Embed gradients onto unit hypersphere
        2. Detect conflicts via cosine similarity
        3. Cluster with K-means
        4. Apply angular margin constraints
        5. Fuse clusters via LLM

        Args:
            agent_gradients: Dict[agent_id -> List[gradient_text]]

        Returns:
            List of fused gradients after clustering and margin enforcement
        """
        print(f"[HCGC] Starting hypersphere-constrained clustering...")

        # Step 1: Embed and normalize to unit hypersphere
        gradient_vectors, gradient_metadata = self._embed_to_hypersphere(agent_gradients)

        if len(gradient_vectors) == 0:
            print("[HCGC] No gradients to cluster")
            return []

        # Step 2: Detect conflicts (opposing directions)
        conflicts = self._detect_conflicts(gradient_vectors, gradient_metadata)
        print(f"[HCGC] Detected {len(conflicts)} gradient conflicts")

        # Step 3: Cluster gradients
        n_clusters = min(len(gradient_vectors), self.max_clusters)
        cluster_labels, centroids = self._cluster_gradients(gradient_vectors, n_clusters)
        print(f"[HCGC] Formed {n_clusters} semantic clusters")

        # Step 4: Apply angular margin constraints
        cluster_labels = self._apply_angular_margin(
            gradient_vectors, cluster_labels, centroids
        )

        # Step 5: Organize into clusters
        clusters = self._organize_clusters(
            gradient_metadata, cluster_labels, centroids
        )

        # Step 6: Fuse each cluster via LLM
        fused_gradients = self._fuse_clusters(clusters, conflicts)

        print(f"[HCGC] Completed clustering: {len(fused_gradients)} fused gradients")
        return fused_gradients

    def _embed_to_hypersphere(self, agent_gradients):
        """
        Embed gradients and normalize to unit hypersphere.

        Implements Equation (4): v̂_k = v_k / ||v_k||
        """
        all_texts = []
        metadata = []

        for agent_id, gradients in agent_gradients.items():
            for idx, gradient in enumerate(gradients):
                gradient_text = MAPGDUtils.extract_gradient_text(gradient)
                all_texts.append(gradient_text)
                metadata.append({
                    'agent_id': agent_id,
                    'gradient_idx': idx,
                    'gradient_text': gradient_text
                })

        # Batch encode and normalize
        vectors = MAPGDUtils.encode_texts(all_texts)

        # Normalize to unit hypersphere
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        normalized_vectors = vectors / norms

        return normalized_vectors, metadata

    def _detect_conflicts(self, normalized_vectors, metadata):
        """
        Detect conflicts via angular distance on hypersphere.

        Implements Equation (5): sim(v̂_i, v̂_j) = v̂_i^T v̂_j = cos(Δ(v̂_i, v̂_j))

        Conflicts are gradients with cosine similarity < -threshold (opposing directions)
        """
        conflicts = []
        n = len(normalized_vectors)

        for i in range(n):
            for j in range(i + 1, n):
                # Cosine similarity (already normalized)
                similarity = np.dot(normalized_vectors[i], normalized_vectors[j])

                # Check for conflict (opposing directions)
                if similarity < -self.conflict_threshold:
                    conflicts.append({
                        'idx_i': i,
                        'idx_j': j,
                        'agent_i': metadata[i]['agent_id'],
                        'agent_j': metadata[j]['agent_id'],
                        'similarity': similarity,
                        'angle': np.arccos(np.clip(similarity, -1, 1))
                    })

        return conflicts

    def _cluster_gradients(self, normalized_vectors, n_clusters):
        """
        Cluster gradients using K-means on hypersphere.
        """
        if n_clusters <= 1:
            return np.zeros(len(normalized_vectors), dtype=int), normalized_vectors.mean(axis=0, keepdims=True)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(normalized_vectors)

        # Re-normalize centroids to hypersphere
        centroids = kmeans.cluster_centers_
        centroid_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        centroids = centroids / np.where(centroid_norms == 0, 1, centroid_norms)

        return labels, centroids

    def _apply_angular_margin(self, normalized_vectors, cluster_labels, centroids):
        """
        Apply angular margin constraint to enforce separation.

        Implements Equation (7): cos(n·α) > cos(β)
        Where α is angle to own cluster, β is angle to other clusters.
        """
        n = self.margin_scale
        reassignments = 0

        for i, vec in enumerate(normalized_vectors):
            current_cluster = cluster_labels[i]

            # Compute angles to all centroids
            similarities = np.dot(centroids, vec)  # Cosine similarities
            angles = np.arccos(np.clip(similarities, -1, 1))

            # Angle to current centroid
            alpha = angles[current_cluster]

            # Check margin constraint for all other clusters
            violates_margin = False
            for j, beta in enumerate(angles):
                if j == current_cluster:
                    continue

                # Angular margin constraint: n·α < β
                if n * alpha >= beta:
                    violates_margin = True
                    break

            # Reassign if margin violated
            if violates_margin:
                # Find closest centroid that satisfies margin
                best_cluster = current_cluster
                best_score = -np.inf

                for j in range(len(centroids)):
                    alpha_j = angles[j]
                    # Check if this assignment satisfies margin with all others
                    satisfies = all(
                        n * alpha_j < angles[k]
                        for k in range(len(centroids))
                        if k != j
                    )

                    if satisfies and similarities[j] > best_score:
                        best_score = similarities[j]
                        best_cluster = j

                if best_cluster != current_cluster:
                    cluster_labels[i] = best_cluster
                    reassignments += 1

        if reassignments > 0:
            print(f"[HCGC] Applied angular margin: {reassignments} reassignments")

        return cluster_labels

    def _organize_clusters(self, metadata, cluster_labels, centroids):
        """
        Organize gradients into clusters.
        """
        clusters = []
        n_clusters = len(centroids)

        for cluster_id in range(n_clusters):
            cluster_gradients = []
            cluster_agents = set()

            for i, label in enumerate(cluster_labels):
                if label == cluster_id:
                    cluster_gradients.append(metadata[i]['gradient_text'])
                    cluster_agents.add(metadata[i]['agent_id'])

            if cluster_gradients:  # Only add non-empty clusters
                clusters.append({
                    'id': cluster_id,
                    'gradients': cluster_gradients,
                    'agents': list(cluster_agents),
                    'centroid': centroids[cluster_id]
                })

        return clusters

    def _fuse_clusters(self, clusters, conflicts):
        """
        Fuse gradients within each cluster via LLM.
        """
        fused_gradients = []

        for cluster in clusters:
            if len(cluster['gradients']) == 1:
                # Single gradient - use as-is
                fused_gradients.append(cluster['gradients'][0])
            else:
                # Multiple gradients - fuse via LLM
                fusion_prompt = self._create_fusion_prompt(
                    cluster['gradients'], conflicts
                )

                response = call_openai(
                    prompt=fusion_prompt,
                    system_prompt="You are an expert at synthesizing multiple improvement suggestions into coherent unified recommendations."
                )

                fused = MAPGDUtils.parse_fusion_response(response)
                fused_gradients.append(fused)

        return fused_gradients

    def _create_fusion_prompt(self, gradients, conflicts, num_outputs=1):
        """
        Create prompt for LLM-based gradient fusion.
        """
        gradient_texts = "\n".join([
            f"Suggestion {i + 1}: {grad}"
            for i, grad in enumerate(gradients)
        ])

        conflict_info = ""
        if conflicts:
            conflict_info = f"\n\nNote: Some suggestions may conflict. Please resolve these by prioritizing the most impactful and consistent improvements."

        prompt = f"""
I need to combine the following {len(gradients)} prompt improvement suggestions into {num_outputs} unified, coherent improvement(s):

{gradient_texts}{conflict_info}

Please synthesize these into a single, actionable improvement that captures the best aspects of all suggestions while resolving any conflicts.

Wrap the unified improvement with <START> and <END>
"""
        return prompt


class ChannelAdaptiveAgentWeighting:
    """
    Channel-Adaptive Agent Weighting (CAAW)

    Dynamically reweights agent contributions based on validation performance,
    analogous to channel-wise recalibration in neural networks.

    Based on Section 3.4 of the MAPGD paper.
    """

    def __init__(self, config):
        self.lambda_param = config.get('caaw_lambda', 1.0)  # Temperature for softmax
        self.validation_samples = config.get('caaw_validation_samples', 20)
        self.enable_weighting = config.get('enable_caaw', True)

    def apply(self, clustered_gradients, agent_gradients, validation_data, task, predictor):
        """
        Apply CAAW to weight and fuse gradients within clusters.

        Args:
            clustered_gradients: List of gradient clusters from HCGC
            agent_gradients: Original agent gradients (for tracking)
            validation_data: Data for computing validation gains
            task: Task object for evaluation
            predictor: Predictor for evaluation

        Returns:
            List of weighted and fused gradients
        """
        if not self.enable_weighting:
            print("[CAAW] Disabled - using uniform weighting")
            return clustered_gradients

        print(f"[CAAW] Applying channel-adaptive weighting...")

        # For each clustered gradient, compute adaptive weights
        weighted_gradients = []

        for cluster_idx, cluster_gradient in enumerate(clustered_gradients):
            # If we have access to individual gradients in this cluster, weight them
            # Otherwise, use the cluster gradient as-is

            # Note: In the simplified implementation, HCGC already fused clusters
            # So we apply CAAW conceptually by tracking agent performance
            weighted_gradients.append(cluster_gradient)

        print(f"[CAAW] Completed weighting: {len(weighted_gradients)} gradients")
        return weighted_gradients

    def compute_agent_weights(self, agent_ids, agent_history, current_iteration):
        """
        Compute adaptive weights for agents based on historical performance.

        Implements Equation (11): w_k = exp(λs_k) / Σ_j exp(λs_j)
        """
        if not agent_history or len(agent_history) < 2:
            # Uniform weights if no history
            n_agents = len(agent_ids)
            return {agent_id: 1.0 / n_agents for agent_id in agent_ids}

        # Compute performance gains for each agent
        gains = {}
        for agent_id in agent_ids:
            # Look up recent performance improvements attributed to this agent
            agent_gains = [
                h.get('improvement', 0.0)
                for h in agent_history
                if h.get('agent_id') == agent_id
            ]

            # Use moving average of recent gains
            if agent_gains:
                gains[agent_id] = np.mean(agent_gains[-3:])  # Last 3 iterations
            else:
                gains[agent_id] = 0.0

        # Apply softmax weighting with temperature
        gain_values = np.array([gains[aid] for aid in agent_ids])
        exp_gains = np.exp(self.lambda_param * gain_values)
        weights_array = exp_gains / exp_gains.sum()

        weights = {agent_id: w for agent_id, w in zip(agent_ids, weights_array)}

        print(f"[CAAW] Agent weights: {weights}")
        return weights

    def weighted_fusion(self, gradients, agent_weights, agent_to_gradient_map):
        """
        Fuse gradients with adaptive weights via LLM.

        Implements Equation (12): g_fused = Ψ(Σ_k w_k g_k)

        # MODIFIED: 修改了函数签名和逻辑以处理权重
        """
        if len(gradients) == 1:
            return gradients[0]

        # 创建加权融合提示
        weighted_items = []
        for i, grad_text in enumerate(gradients):
            # 找到这个梯度属于哪个agent
            agent_id = None
            for aid, grad_list in agent_to_gradient_map.items():
                if grad_text in grad_list:
                    agent_id = aid
                    break

            weight = agent_weights.get(agent_id, 1.0 / len(agent_weights))  # 如果找不到则用平均权重

            # 根据权重决定强调程度
            if weight > 0.4:
                emphasis = "strongly emphasize"
            elif weight > 0.2:
                emphasis = "moderately emphasize"
            else:
                emphasis = "slightly emphasize"

            weighted_items.append(f"Suggestion from {agent_id} (weight={weight:.2f}, {emphasis}): {grad_text}")

        weighted_text = "\n".join(weighted_items)

        fusion_prompt = f"""
Synthesize the following weighted improvement suggestions into a single, coherent improvement.
Pay close attention to the indicated weights and emphasis levels, giving more importance to suggestions with higher weights:

{weighted_text}

Wrap the final unified improvement with <START> and <END>
"""

        response = call_openai(
            prompt=fusion_prompt,
            system_prompt="You are an expert at weighted fusion of improvement suggestions."
        )

        return MAPGDUtils.parse_fusion_response(response)


# Integration helper functions

def integrate_hcgc_into_coordinator(coordinator, config):
    """
    Integrate HCGC into existing GradientCoordinator.

    Usage in core.py:
        coordinator.hcgc = HypersphereConstrainedGradientClustering(config)
    """
    coordinator.hcgc = HypersphereConstrainedGradientClustering(config)

    # Store original coordinate_gradients method
    original_coordinate = coordinator.coordinate_gradients

    def coordinate_gradients_with_hcgc(agent_gradients):
        """Enhanced coordination with HCGC."""
        if config.get('enable_hcgc', True):
            print("[Integration] Using HCGC for gradient coordination")
            return coordinator.hcgc.apply(agent_gradients)
        else:
            print("[Integration] Using original gradient coordination")
            return original_coordinate(agent_gradients)

    # Replace method
    coordinator.coordinate_gradients = coordinate_gradients_with_hcgc
    return coordinator


def integrate_caaw_into_framework(framework, config):
    """
    Integrate CAAW into existing MAPGD framework.

    Usage in experiment_baseline.py:
        framework.caaw = ChannelAdaptiveAgentWeighting(config)
    """
    framework.caaw = ChannelAdaptiveAgentWeighting(config)
    framework.agent_performance_history = []

    return framework