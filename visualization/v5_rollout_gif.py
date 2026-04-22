"""
V5 高层策略 rollout → GIF + 物理约束核查
用法:
  python visualization/v5_rollout_gif.py \
      --ckpt checkpoints/v5_td3bc_B/latest.pth \
      --output td3bc_B.gif \
      --episodes 5
"""
import os, sys, argparse, numpy as np, torch
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from high_tabular.hrl_v3_real_auv_direct_thrust import (
    PursuitEnvDirectThrust, load_v5_nn)
from training.offline_ppo_pretrain import HighLevelActorCritic
from visualization.hq_pursuit_gif import (
    PursuitGIFGenerator, collect_trajectory_from_env)


def build_env(device, args):
    low = load_v5_nn(args.low_level_ckpt, device)
    return PursuitEnvDirectThrust(
        nn_policy=low, device=device,
        world_size=args.world_size, catch_radius=args.catch_radius,
        v_max_pursuer=args.v_max_pursuer,
        a_max_pursuer=args.a_max_pursuer,
        v_max_evader=args.v_max_evader,
        a_max_evader=args.a_max_evader,
        seed=args.seed,
    )


def load_policy(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    pol = HighLevelActorCritic(obs_dim=21, act_dim=2, hidden=(256, 128))
    if "model" in ckpt:
        pol.load_state_dict(ckpt["model"], strict=False)
        src = "model"
    elif "actor" in ckpt:
        pol.load_state_dict(ckpt["actor"], strict=False)
        src = "actor"
    else:
        raise ValueError(f"Unknown ckpt keys: {list(ckpt.keys())}")
    pol.to(device).eval()
    return pol, src


def analyze(traj, v_max_p, a_max_p, v_max_e, a_max_e):
    """核查速度/加速度是否超限。"""
    # 环境 step 之间的加速度是 "一个 high-step 内的平均加速度"（dt=low_steps*dt_low）。
    # 更保真的是低频测量（此处 dt = subgoal 周期）。阈值比较用这个 dt 下的极值。
    p_spd = traj.get_pursuer_speeds()
    e_spd = traj.get_evader_speeds()
    p_acc = traj.get_pursuer_accs()
    e_acc = traj.get_evader_accs()
    print(f"  pursuer: speed max={p_spd.max():.3f} (limit {v_max_p})  "
          f"| acc max={p_acc.max():.3f} (limit {a_max_p})")
    print(f"  evader : speed max={e_spd.max():.3f} (limit {v_max_e})  "
          f"| acc max={e_acc.max():.3f} (limit {a_max_e})")
    # 环境在 inner 物理步会钳位；这里是 subgoal 间隔的平均数，通常 < limit
    v_ok_p = p_spd.max() <= v_max_p * 1.01
    v_ok_e = e_spd.max() <= v_max_e * 1.01
    return dict(v_ok_p=v_ok_p, v_ok_e=v_ok_e,
                p_spd_max=float(p_spd.max()),
                e_spd_max=float(e_spd.max()),
                p_acc_max=float(p_acc.max()),
                e_acc_max=float(e_acc.max()))


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--ckpt", required=True)
    pa.add_argument("--low-level-ckpt", default="low_near/v5_direct_thrust/checkpoints/best.pth")
    pa.add_argument("--output", default="v5_rollout.gif")
    pa.add_argument("--episodes", type=int, default=3,
                    help="尝试的 episode 数；会选一个成功的 + 最长的")
    pa.add_argument("--max-steps", type=int, default=6000)
    pa.add_argument("--world-size", type=float, default=30.0)
    pa.add_argument("--catch-radius", type=float, default=1.0)
    pa.add_argument("--v-max-pursuer", type=float, default=0.65)
    pa.add_argument("--a-max-pursuer", type=float, default=0.5)
    pa.add_argument("--v-max-evader", type=float, default=0.85)
    pa.add_argument("--a-max-evader", type=float, default=0.25)
    pa.add_argument("--seed", type=int, default=42)
    pa.add_argument("--fps", type=int, default=20)
    pa.add_argument("--skip", type=int, default=2)
    args = pa.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    pol, src = load_policy(args.ckpt, device)
    print(f"Policy: {args.ckpt}  (key={src})")

    trajs = []
    for ep in range(args.episodes):
        args_ep = argparse.Namespace(**{**vars(args), "seed": args.seed + ep})
        env = build_env(device, args_ep)
        traj = collect_trajectory_from_env(env, pol, device,
                                           deterministic=True,
                                           max_steps=args.max_steps)
        print(f"[ep {ep}] steps={traj.n_steps}  success={traj.success}  "
              f"final_dist={traj.distances[-1]:.2f}")
        stats = analyze(traj, args.v_max_pursuer, args.a_max_pursuer,
                        args.v_max_evader, args.a_max_evader)
        trajs.append((traj, stats))

    # 挑一个: 优先 success 最短；否则最长
    success = [(t,s) for t,s in trajs if t.success]
    if success:
        traj, stats = min(success, key=lambda x: x[0].n_steps)
        tag = f"SUCCESS steps={traj.n_steps}"
    else:
        traj, stats = max(trajs, key=lambda x: x[0].n_steps)
        tag = f"FAIL steps={traj.n_steps}"
    print(f"\nPicked: {tag}")

    gen = PursuitGIFGenerator(
        world_size=args.world_size, catch_radius=args.catch_radius,
        v_max_pursuer=args.v_max_pursuer, v_max_evader=args.v_max_evader,
        a_max_pursuer=args.a_max_pursuer, a_max_evader=args.a_max_evader)
    title = f"V5 {os.path.basename(args.ckpt)}  {tag}"
    gen.generate_gif(traj, args.output, fps=args.fps,
                     skip_frames=args.skip, title=title)

    # 约束总览 (全部 ep)
    print("\n=== 约束核查（所有 episodes）===")
    print(f"v_max_pursuer 限制 = {args.v_max_pursuer:.2f} m/s")
    print(f"a_max_pursuer 限制 = {args.a_max_pursuer:.2f} m/s²")
    print(f"v_max_evader  限制 = {args.v_max_evader:.2f} m/s")
    print(f"a_max_evader  限制 = {args.a_max_evader:.2f} m/s²")
    all_p = np.concatenate([t.get_pursuer_speeds() for t,_ in trajs])
    all_e = np.concatenate([t.get_evader_speeds()  for t,_ in trajs])
    print(f"pursuer speed: max={all_p.max():.3f}  mean={all_p.mean():.3f}")
    print(f"evader  speed: max={all_e.max():.3f}  mean={all_e.mean():.3f}")


if __name__ == "__main__":
    main()
