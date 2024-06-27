from agent import Subject, Sync

subj = Subject()
subj.run_active_inference(15)
subj.plot_trajectory()

sync = Sync()
sync.run_active_inference(5)
sync.plot_trajectories()
