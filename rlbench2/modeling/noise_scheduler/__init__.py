from .rectified_flow import RFScheduler


def fetch_schedulers(denoise_model, denoise_timesteps):
    if denoise_model != "rectified_flow":
        raise ValueError(f"Unsupported RLBench scheduler: {denoise_model}")
    return RFScheduler(), RFScheduler()
