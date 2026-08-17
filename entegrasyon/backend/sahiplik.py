# -*- coding: utf-8 -*-
"""Sahiplik (operatör ülke) çözücü.

VRAG tarafından tanınan model adını, LLM karar-destek katmanının platform
kayıtları + Türkiye envanteri verisiyle eşleştirip "sahiplik" bilgisini üretir:

  - menşei  : modelin köken/üretici ülkesi (VRAG metadata'sındaki `ulke`) — burada
              hesaplanmaz, adapter mevcut alandan geçirir.
  - sahiplik: modelin Türkiye envanterinde (turkey_inventory.json) olup olmamasına
              göre "Türkiye" (dost/envanterde) / "Yabancı" (envanterde değil) /
              "Bilinmiyor" (platform kaydında hiç eşleşmedi).

Eşleştirme, LLM alt-sisteminin `platform_aliases.json` + `platform_registry.json`
alias tablolarıyla yapılır (ad normalize edilerek: küçük harf, alfanümerik dışı
karakterler atılarak). Bu modül yalnızca birkaç küçük JSON'u başlangıçta bir kez
okur; runtime'da sadece sözlük araması yapar (hızlı, LLM servisine çağrı yok).
"""
import json
import re
from pathlib import Path

# entegrasyon/backend/ -> proje kökü
_ROOT = Path(__file__).resolve().parent.parent.parent
_LLM_DATA = _ROOT / "LLM" / "data"


def _norm(s: str) -> str:
    """Ad karşılaştırması için normalize et: küçük harf + yalnız alfanümerik."""
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


class SahiplikCozucu:
    def __init__(self):
        self._alias_to_pid: dict[str, str] = {}
        self._tr_pids: set[str] = set()
        self._yukle()

    def _yukle(self) -> None:
        # 1) Alias tablosu: alias -> platform_id
        try:
            ali = json.loads(
                (_LLM_DATA / "platforms" / "platform_aliases.json").read_text(encoding="utf-8")
            )
            for a in ali.get("aliases", []):
                if a.get("alias") and a.get("platform_id"):
                    self._alias_to_pid.setdefault(_norm(a["alias"]), a["platform_id"])
        except (OSError, ValueError):
            pass

        # 2) Registry canonical_name + aliases (ek kapsama)
        try:
            reg = json.loads(
                (_LLM_DATA / "platforms" / "platform_registry.json").read_text(encoding="utf-8")
            )
            for p in reg.get("platforms", []):
                pid = p.get("platform_id")
                if not pid:
                    continue
                if p.get("canonical_name"):
                    self._alias_to_pid.setdefault(_norm(p["canonical_name"]), pid)
                for al in p.get("aliases", []) or []:
                    self._alias_to_pid.setdefault(_norm(al), pid)
        except (OSError, ValueError):
            pass

        # 3) Türkiye envanteri: aktif platform_id kümesi
        try:
            inv = json.loads(
                (_LLM_DATA / "inventory" / "turkey_inventory.json").read_text(encoding="utf-8")
            )
            self._tr_pids = {
                r["platform_id"]
                for r in inv.get("records", [])
                if r.get("active") and r.get("platform_id")
            }
        except (OSError, ValueError):
            pass

    def platform_id(self, model_adi: str) -> str | None:
        """VRAG model adını platform_id'ye çözer (yoksa None)."""
        if not model_adi:
            return None
        return self._alias_to_pid.get(_norm(model_adi))

    def sahiplik(self, model_adi: str) -> str:
        """'Türkiye' (envanterde) / 'Yabancı' (envanterde değil) / 'Bilinmiyor' (eşleşmedi)."""
        pid = self.platform_id(model_adi)
        if pid is None:
            return "Bilinmiyor"
        return "Türkiye" if pid in self._tr_pids else "Yabancı"

    @property
    def hazir(self) -> bool:
        return bool(self._alias_to_pid) and bool(self._tr_pids)
