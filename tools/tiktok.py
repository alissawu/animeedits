# Download a TikTok video (no watermark) via the tikwm.com API.
# usage: python tools/tiktok.py <tiktok url> [out.mp4]
import sys, httpx

def download(url, out=None):
    r = httpx.get("https://www.tikwm.com/api/", params={"url": url, "hd": 1}, timeout=30).json()
    if r.get("code") != 0:
        raise SystemExit(f"tikwm error: {r.get('msg')}")
    d = r["data"]
    src = d.get("hdplay") or d.get("play")
    out = out or f"tiktok_{d['id']}.mp4"
    with httpx.stream("GET", src, follow_redirects=True, timeout=120) as s, open(out, "wb") as f:
        for chunk in s.iter_bytes():
            f.write(chunk)
    print(out)
    return out

if __name__ == "__main__":
    download(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
