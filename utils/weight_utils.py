import torch


def load_vitpose_pretrained(model: torch.nn.Module, checkpoint_path: str):
    """Load ViTPose pre-trained weights into ``model``.

    Parameters whose shapes do not match will be skipped so that the load does
    not fail.  This allows loading ViTPose weights even when the architecture of
    ``model`` differs from the original pose model (e.g. different patch size or
    embedding dimension).
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model_state = model.state_dict()
    filtered_state = {}
    ignored = 0

    for key, value in state_dict.items():
        original_key = key
        if key.startswith("module."):
            key = key[len("module."):]

        if not key.startswith("backbone."):
            key = f"backbone.{key}"

        if key in model_state and model_state[key].shape == value.shape:
            filtered_state[key] = value
        else:
            ignored += 1

    missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    print(f"Loaded {len(filtered_state)} params from pretrained weights; ignored {ignored} mismatched params.")
    return missing, unexpected