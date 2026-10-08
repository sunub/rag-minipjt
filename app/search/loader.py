"""Fetch the pinned law; never save the authenticated request URL."""
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx

from common.config import SERVICE

SNAPSHOT_DATE = '2026-07-21'
EXPECTED_PROMULGATION = '21311'


def validate_root(root):
    if root.tag != '법령' or root.find('./조문/조문단위') is None:
        raise ValueError('법령 API가 유효한 법령 XML을 반환하지 않았습니다.')
    if root.findtext('./기본정보/공포번호') != EXPECTED_PROMULGATION:
        raise ValueError('평가 대상은 제21311호입니다. 법령 버전을 확인하세요.')
    return root


async def fetch_law_root(cache_path=None, refresh=False):
    path = Path(cache_path) if cache_path else None
    if path and path.exists() and not refresh:
        return validate_root(ET.parse(path).getroot())
    if not SERVICE.base_url or not SERVICE.oc_key:
        raise ValueError('SERVICE_BASE_URL과 OC_KEY를 설정하세요.')
    async with httpx.AsyncClient(timeout=45, follow_redirects=True) as client:
        response = await client.get(str(SERVICE.base_url), params={
            'OC': SERVICE.oc_key, 'target': SERVICE.target, 'MST': SERVICE.mst, 'type': 'XML',
        })
        response.raise_for_status()
    root = validate_root(ET.fromstring(response.content))
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        ET.ElementTree(root).write(path, encoding='utf-8', xml_declaration=True)
    return root
