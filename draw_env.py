import matplotlib.pyplot as plt
from envs.MA_UCMEC_dyna_noncoop_small import MA_UCMEC_dyna_noncoop

env = MA_UCMEC_dyna_noncoop(seed=884)
env.reset()

users = env.locations_users[:env.M_sim]
aps = env.locations_aps[:env.N_sim]
cpus = env.locations_cpu

plt.figure(figsize=(6, 6))
plt.scatter(aps[:, 0], aps[:, 1], s=30, c="tab:blue", label="AP")
plt.scatter(users[:, 0], users[:, 1], s=30, c="tab:orange", label="User")
plt.scatter(cpus[:, 0], cpus[:, 1], s=120, c="tab:green", marker="^", label="CPU")
plt.xlim(0, 300)
plt.ylim(0, 300)
plt.gca().set_aspect("equal", adjustable="box")
plt.legend()
plt.title("AP / User / CPU Locations")
plt.show()

