import wandb
import datetime
import ml_collections
def set_resume_info(config):
    """
        Resume wandb training log
    """
    project_name = config.project_name
    run_id = config.resume_from_id
    # Get history
    api_run = wandb.Api().run(f"{project_name}/{run_id}")
    history = api_run.history()
    if not history.empty:
        config.resume_from_step = int(history['_step'].iloc[-1])
        config.resume_from_epoch = int(history['epoch'].iloc[-1])
        print(f"Auto-resuming from step {config.resume_from_step}, epoch {config.resume_from_epoch}")
    else:
        print("No previous history found, starting from beginning")
        config.resume_from_step = 0
        config.resume_from_epoch = 0

    config.run_name = api_run.name
    config.run_id = api_run.id

def set_wandb(config):
    # Resume training
    if config.resume_from_id:
        run_id = config.resume_from_id
        print("Resuming")
        wandb_run = wandb.init(
            project=config.project_name,
            config=config.to_dict(),
            id=run_id,
            resume='must'
        )
        print("Resuming done")
    else:
        print("Start from beginning")
        unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
        if not config.run_name:
            config.run_name = unique_id
        else:
            config.run_name += "_" + unique_id

        wandb_run = wandb.init(
            project=config.project_name,
            config=config.to_dict()
        )
        config.run_name = wandb_run.name
        config.run_id = wandb_run.id
        print("Init done")

    return wandb_run


config = ml_collections.ConfigDict()

config.resume_from_id = 'd04p82m0'
config.project_name = 'FlowGRPO'
set_resume_info(config)
set_wandb(config)