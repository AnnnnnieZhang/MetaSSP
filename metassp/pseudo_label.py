"""SDF-aware pseudo-label weighting (paper Sec. III, Eq. 4-5)."""


def pseudo_label_weight(consistency_loss, sdf_variance, alpha=4.0, beta=4.0):
    """Reliability of a batch of pseudo-labels (paper Eq. 4).

        w_pseudo = clip(1 - alpha * l_cons - beta * sigma^2, 0, 1)

    Args:
        consistency_loss: loss between the teacher predictions under strong and
            weak augmentation (float).
        sdf_variance: variance of the teacher SDF under weak augmentation (float).
        alpha, beta: penalties on inconsistency and SDF variance.
    """
    w = 1.0 - alpha * consistency_loss - beta * sdf_variance
    return max(0.0, min(w, 1.0))


def blend_losses(supervised_loss, pseudo_loss, w_pseudo, lam=0.2):
    """Blend supervised and pseudo-supervised loss terms (paper Eq. 5).

        L_k = (1 - lam * w_pseudo) * L_sup,k + lam * w_pseudo * L_unsup,k

    Args:
        supervised_loss, pseudo_loss: dicts of loss terms sharing the same keys.
        w_pseudo: pseudo-label weight in [0, 1].
        lam: maximum contribution of the pseudo-labels.
    """
    w = lam * w_pseudo
    return {key: (1 - w) * supervised_loss[key] + w * pseudo_loss[key] for key in supervised_loss}
