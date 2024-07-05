from agent import Subject, Sync, Three

subj = Subject()
subj.run_active_inference(15)
subj.plot_trajectory()

sync = Sync()
sync.run_active_inference(15)
sync.plot_trajectories()

three = Three()
three.run_active_inference(15)
three.plot_trajectories()
