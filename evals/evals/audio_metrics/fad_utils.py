from frechet_audio_distance import FrechetAudioDistance

def make_fad_clap(sample_rate: int = 48000, submodel: str = "630k-audioset", enable_fusion: bool = False, verbose: bool = False):
    return FrechetAudioDistance(
        model_name="clap",
        sample_rate=sample_rate,
        submodel_name=submodel,
        verbose=verbose,
        enable_fusion=enable_fusion,
    )
