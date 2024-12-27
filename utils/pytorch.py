def load_latest_models(save_path, actor, reward_predictor, device):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        logger.info(f"Created save directory at {save_path}")
        return
    folders = [f for f in os.listdir(save_path) if os.path.isdir(os.path.join(save_path, f)) and "_" in f]
    if not folders:
        logger.info("No saved models found. Starting from scratch.")
        return
    try:
        latest_folder = max(folders, key=lambda f: datetime.strptime(f.split('_')[-1], "%d%m%y_%H%M%S"))
        actor_path = os.path.join(save_path, latest_folder, "actor.pth")
        reward_predictor_path = os.path.join(save_path, latest_folder, "reward_predictor.pth")
        if os.path.exists(actor_path):
            actor.load_state_dict(torch.load(actor_path, map_location=device))
            logger.info(f"Loaded actor from {actor_path}")
        if os.path.exists(reward_predictor_path):
            reward_predictor.load_state_dict(torch.load(reward_predictor_path, map_location=device))
            logger.info(f"Loaded reward predictor from {reward_predictor_path}")
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        logger.info("Starting from scratch.")