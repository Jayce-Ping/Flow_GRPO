import wandb

# ---- 配置信息 ----
run_id = "d04p82m0"
entity_name = "315229706-xi-an-jiaotong-university-"
project_name = "FlowGRPO"

# ---- 恢复运行 ----
run = wandb.init(
    entity=entity_name,
    project=project_name,
    id=run_id,
    resume="must"   # 强制 resume
)

print("Run name:", run.name)
print("Run id:", run.id)

# ---- 用 API 获取历史 ----
api = wandb.Api()
api_run = api.run(f"{entity_name}/{project_name}/{run_id}")

history = api_run.history()   # pandas.DataFrame
print("History head:\n", history.head())

# ---- 取最后一个 step ----
resume_info = {}
if not history.empty:
    last_step = history.index.max()   # history 的 index 默认是 step
    resume_info['global_step'] = int(last_step)

print("Resume info:", resume_info)
