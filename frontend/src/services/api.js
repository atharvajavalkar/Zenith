import axios from 'axios'

const API_BASE = import.meta.env.VITE_API_URL || ''

let accessToken = localStorage.getItem('algo_access_token')
let refreshToken = localStorage.getItem('algo_refresh_token')
let onUnauthorized = null

export function setAccessToken(token) {
  accessToken = token
}

export function setRefreshToken(token) {
  refreshToken = token
}

export function setOnUnauthorized(fn) {
  onUnauthorized = fn
}

const api = axios.create({
  baseURL: API_BASE,
  headers: { 'Content-Type': 'application/json' },
})

api.interceptors.request.use((config) => {
  if (accessToken) {
    config.headers.Authorization = `Bearer ${accessToken}`
  }
  return config
})

api.interceptors.response.use(
  (res) => res,
  async (error) => {
    const original = error.config
    if (error.response?.status === 401 && !original._retry && refreshToken) {
      original._retry = true
      try {
        const { data } = await axios.post(`${API_BASE}/api/auth/refresh`, {
          refresh_token: refreshToken,
        })
        accessToken = data.access_token
        refreshToken = data.refresh_token
        localStorage.setItem('algo_access_token', data.access_token)
        localStorage.setItem('algo_refresh_token', data.refresh_token)
        original.headers.Authorization = `Bearer ${data.access_token}`
        return api(original)
      } catch {
        if (onUnauthorized) onUnauthorized()
      }
    }
    return Promise.reject(error)
  },
)

export default api
