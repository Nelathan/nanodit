"""How UsuiTrack treats nanodit's parameter shapes (archive 2026-11-01 row 2, PLAN A2), on toy problems; prints, writes nothing."""
import torch
from usuitrack import UsuiTrack

torch.manual_seed(0)

# A zero-init matrix next to a fan-in one: does it move?
final = torch.nn.Parameter(torch.zeros(128, 512))
hidden = torch.nn.Parameter(torch.randn(512, 512) / 512**0.5)
opt = UsuiTrack([final, hidden], lr=1e-2)
x = torch.randn(32, 512)
for step in range(5):
    ((x @ hidden.T) @ final.T - 1).square().mean().backward()
    opt.step(), opt.zero_grad()
    print(f"zero-init: step {step} final norm {final.norm():.4f}, hidden norm {hidden.norm():.4f}")

# The shared adaLN projection: gate rows zero inside a matrix whose shift/scale rows are small random.
w = torch.randn(6, 64, 64) * 0.02
w[[2, 5]] = 0
w = torch.nn.Parameter(w.reshape(6 * 64, 64))
opt = UsuiTrack([w], lr=1e-2)
c, target = torch.randn(16, 64), torch.randn(16, 6 * 64)
for step in range(5):
    (c @ w.T - target).square().mean().backward()
    opt.step(), opt.zero_grad()
    rows = w.detach().view(6, 64, 64)
    print(f"partial-zero: step {step} gate rows {rows[[2, 5]].norm():.4f}, others {rows[[0, 1, 3, 4]].norm():.4f}, matrix {w.norm():.4f}")

# A 4-D and a 1-D param.
for p in (torch.nn.Parameter(torch.zeros(512, 1, 3, 3)), torch.nn.Parameter(torch.ones(64))):
    try:
        UsuiTrack([p])
    except ValueError as e:
        print(f"{tuple(p.shape)} rejected: {e}")

# A per-block modulation table [6, 512].
table = torch.nn.Parameter(torch.randn(6, 512) * 0.02)
opt = UsuiTrack([table])
table.square().sum().backward()
opt.step()
print(f"table {tuple(table.shape)} accepted: frame {tuple(opt.state[table]['frame'].shape)}")
