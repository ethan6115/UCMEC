import matplotlib.pyplot as plt
from envs.ucmec_hierarchical import UCMEC_hierarchical_env



env = UCMEC_hierarchical_env(seed=1)
env.reset()

users = env.locations_users[:env.M_sim]
aps = env.locations_aps[:env.N_sim]
cpus = env.locations_cpu

plt.figure(figsize=(6, 6))
plt.scatter(aps[:, 0], aps[:, 1], s=22, c="tab:blue", marker="o", label="AP")
plt.scatter(users[:, 0], users[:, 1], s=42, c="tab:orange", marker="D", label="User")
plt.scatter(cpus[:, 0], cpus[:, 1], s=120, c="tab:green", marker="^", label="CPU")
plt.xlabel("X coordinate (m)")
plt.ylabel("Y coordinate (m)")
plt.xlim(0, 900)
plt.ylim(0, 900)
plt.gca().set_aspect("equal", adjustable="box")
plt.legend()
plt.show()
