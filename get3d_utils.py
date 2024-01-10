import clip
import torch
from torch.nn.functional import normalize

from submodules.CLIP_PAE.utils import get_embeddings_from_text_file, gram_schmidt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from os import path


# given batches of image features I and text features T,
# return targets (len(targets)=len(T)*len(I)) and the subspace basis
# targets[i*len(I) + j] is the projected embedding from T[i] and I[j]
@torch.no_grad()
def get_pae(model, n_components, image_features, text_features, attribute='car_type', power=9.5):
    basis = torch.eye(text_features.shape[1], device='cuda')

    # projection
    subspace_basis = get_pae_PCA_basis(n_components=n_components, attribute=attribute)
    text_coeff_sum = (text_features @ subspace_basis.T).sum(dim=-1)
    image_coeff = image_features @ subspace_basis.T
    image_coeff = power * abs(image_coeff)

    # augmentation
    targets = []
    for i, text_feature in enumerate(text_features):
        for j, image_feature in enumerate(image_features):
            target = image_feature.clone()
            # if "Ex" in args.target:
            #     target = image_feature + args.power * text_feature - to_deduct_per_img[j]
            # elif "+" in args.target:
            #     shift = 0
            #     for k, basis_vector in enumerate(subspace_basis):
            #         coeff = image_coeff[j][k]
            #         shift += coeff
            #         target -= coeff * basis_vector
            #     target += shift / text_coeff_sum[i] * text_feature
            # else: # pae
            target += power * text_feature
            targets.append(target)
    targets = torch.stack(targets)
    return targets, subspace_basis

@torch.no_grad()
def get_pae_PCA_basis(n_components=10, attribute="car_type"):
    basis_path = f"data/corpus/{attribute}_space_basis_{n_components}.pt"
    if path.exists(basis_path):
        return torch.load(basis_path, map_location='cuda')

    # No precomputed basis. Compute now
    all_embeddings = get_embeddings_from_text_file(f"data/corpus/{attribute}")
    type_before = all_embeddings.dtype
    all_embeddings = StandardScaler().fit_transform(all_embeddings.cpu().numpy())
    pca = PCA(n_components=n_components)
    pca.fit(all_embeddings)
    basis = torch.from_numpy(pca.components_).to('cuda')
    torch.save(basis, basis_path)
    return basis