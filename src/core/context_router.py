# -*- coding: utf-8 -*-
"""Bağlam yönlendirici — tespit edilen platforma göre LLM video_id (senaryo bağlamı) seçer.

SORUN (arkadaş revizesi): pipeline operasyonel karar sistemine HER ZAMAN sabit
video_id="live_video" gönderiyordu. live_video bağlamının beklenen platformu
PLT_KAAN olduğundan, F-16 / WZ-7 gibi başka bir platform doğru tespit edilse bile
izin / uçuş planı / NOTAM sorguları YANLIŞ bağlamda yapılıyor ("izin doğrulanamadı",
"uçuş planı bulunamadı"). LLM tarafında zaten hazır olan 94-platformluk yönlendirme
tablosu (raw_vlm_context_routes.json: platform_id → video_id) hiç kullanılmıyordu.

Bu modül o tabloyu devreye alır: VRAG'ın tanıdığı model adı → platform_id (alias
tablosu) → video_id (route tablosu). Eşleşme yoksa fallback_video_id kullanılır.
Yalnızca başlangıçta birkaç küçük JSON okunur; runtime'da sözlük araması yapılır.
"""
import json
import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)

# src/core/ -> proje kökü -> LLM/data
_ROOT = Path(__file__).resolve().parent.parent.parent
_LLM_DATA = _ROOT / "LLM" / "data"


def _norm(s: object) -> str:
    """Ad karşılaştırması için normalize et: küçük harf + yalnız alfanümerik."""
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


class ContextRouter:
    def __init__(self):
        self._alias_to_pid: dict[str, str] = {}
        self._pid_to_video: dict[str, str] = {}
        self._fallback: str | None = None
        self._yukle()

    def _yukle(self) -> None:
        # 1) Alias tablosu: model adı -> platform_id
        try:
            ali = json.loads(
                (_LLM_DATA / "platforms" / "platform_aliases.json").read_text(encoding="utf-8")
            )
            for a in ali.get("aliases", []):
                if a.get("alias") and a.get("platform_id"):
                    self._alias_to_pid.setdefault(_norm(a["alias"]), a["platform_id"])
        except (OSError, ValueError):
            pass
        # Registry canonical_name + aliases (ek kapsama)
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

        # 2) Route tablosu: platform_id -> video_id (+ fallback)
        try:
            routes = json.loads(
                (_LLM_DATA / "seeds" / "raw_vlm_context_routes.json").read_text(encoding="utf-8")
            )
            self._pid_to_video = dict(routes.get("routes", {}))
            self._fallback = routes.get("fallback_video_id")
        except (OSError, ValueError):
            pass

        log.info(
            f"[ROUTER] {len(self._alias_to_pid)} alias, {len(self._pid_to_video)} route "
            f"yüklendi (fallback={self._fallback})."
        )

    def platform_id(self, model_adi: str | None) -> str | None:
        if not model_adi:
            return None
        return self._alias_to_pid.get(_norm(model_adi))

    def resolve_video_id(self, model_adi: str | None) -> str | None:
        """Model adından video_id (senaryo bağlamı) çözer; yoksa fallback döner."""
        pid = self.platform_id(model_adi)
        if pid and pid in self._pid_to_video:
            return self._pid_to_video[pid]
        return self._fallback

    @property
    def hazir(self) -> bool:
        return bool(self._pid_to_video) and self._fallback is not None
