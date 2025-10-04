
from typing import Tuple, Literal
Mode = Literal["DOMINATES", "IS_DOMINATED"]

def infinity_ledger_from_pmfs(mX: float, pnegX: float, pposX: float, mY: float, pnegY: float, pposY: float, mode: Mode) -> Tuple[float, float]:
    add_neg = pnegX * (mY + pnegY) + pnegY * mX
    add_pos = pposX * (mY + pposY) + pposY * mX
    m_cross = pnegX * pposY + pposX * pnegY
    if mode == "DOMINATES":
        add_pos += m_cross
    else:
        add_neg += m_cross
    return float(add_neg), float(add_pos)
