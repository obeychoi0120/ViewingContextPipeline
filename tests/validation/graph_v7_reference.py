"""Small loop oracle: raw edges, per-Action roles and per-Scene means."""

import numpy as np
import torch

from graph_reference import graph_batch as raw_graph_batch
from validation.graph_model import GraphSASRec as PackedSASRec, SceneRoleEncoder as PackedEncoder


def graph_batch(store, item_ids, device):
    batch = raw_graph_batch(store, item_ids, device)
    title_ids = store["titles"][item_ids]
    batch["titles"] = torch.as_tensor(np.array(store["features"][title_ids]), device=device)
    batch["title_available"] = torch.as_tensor((title_ids != 0).astype(np.int64), device=device)
    counts = np.diff(batch["offsets"])
    batch["scene_videos"] = torch.as_tensor(np.repeat(np.arange(len(item_ids)), counts), device=device)
    batch["available"] = torch.as_tensor(((title_ids != 0) | (counts > 0)).astype(np.int64), device=device)
    return batch


class SceneRoleEncoder(PackedEncoder):
    # Share parameter declarations only; the computation uses no packed role
    # matrices, scatter reductions or production forward implementation.
    def forward(self, batch):
        titles = self.title_projection(batch["titles"])
        titles = titles.masked_fill(~batch["title_available"].bool()[:, None], 0)
        videos = []
        zero = titles.new_zeros(self.hidden_dim)
        for video, (start, end) in enumerate(zip(batch["offsets"][:-1], batch["offsets"][1:])):
            scenes = []
            for scene in range(start, end):
                nodes = torch.where(batch["node_scenes"] == scene)[0].tolist()
                entities = {node: self.entity_projection(batch["features"][node])
                            for node in nodes if batch["types"][node] == 0}
                actions = []
                for node in nodes:
                    if batch["types"][node] != 1:
                        continue
                    action = self.action_projection(batch["features"][node])
                    roles = []
                    for slot, role in enumerate(self.ROLE_INDICES):
                        edges = batch["edges"]
                        links = edges[(edges[:, 1] == node) & (edges[:, 2] == role)]
                        if len(links):
                            roles.append(entities[int(links[0, 0])])
                        else:
                            rows = batch["missing"][batch["missing"][:, 0] == node]
                            state = int(rows[0, role + 1]) if len(rows) else 0
                            roles.append(self.role_states[slot, max(state, 0)])
                    actions.append(action + torch.relu(self.role_fusion(torch.cat([action, *roles]))))
                entity_mean = torch.stack(list(entities.values())).mean(0) if entities else zero
                action_mean = torch.stack(actions).mean(0) if actions else zero
                context = self.context_projection(batch["contexts"][scene])
                scenes.append(self.scene_mlp(torch.cat([titles[video], entity_mean, action_mean, context])))
            if scenes:
                videos.append(torch.stack(scenes).mean(0))
            else:
                videos.append(self.scene_mlp(torch.cat([titles[video], zero, zero, zero])))
        return torch.stack(videos).masked_fill(~batch["available"].bool()[:, None], 0)


class GraphSASRec(PackedSASRec):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.graph_encoder = SceneRoleEncoder(
            self.store["features"].shape[1], self.title_dim, kwargs["aggregation"]
        )

    def _calculate(self, ids):
        if not len(ids):
            return self.title_projection.weight.new_empty((0, self.embedding_dim))
        batch = graph_batch(self.store, np.asarray(ids) - 1, self.title_projection.weight.device)
        return self.graph_encoder(batch)
