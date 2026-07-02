const WS_BASE = import.meta.env.VITE_WS_URL || (
  typeof window !== 'undefined'
    ? `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}`
    : 'ws://localhost:8000'
)

export function connectSignalWs(accessToken, { onMessage, onClose }) {
  const url = `${WS_BASE}/ws/signals?token=${encodeURIComponent(accessToken)}`
  const ws = new WebSocket(url)

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data)
      onMessage?.(data)
    } catch {
      // ignore malformed
    }
  }

  ws.onclose = () => onClose?.()

  const pingInterval = setInterval(() => {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send('ping')
    }
  }, 25000)

  return () => {
    clearInterval(pingInterval)
    ws.close()
  }
}
