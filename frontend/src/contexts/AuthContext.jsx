import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react'
import api, { setAccessToken } from '../services/api'

const AuthContext = createContext(null)

const ACCESS_KEY = 'algo_access_token'
const REFRESH_KEY = 'algo_refresh_token'

export function AuthProvider({ children }) {
  const [accessToken, setToken] = useState(() => localStorage.getItem(ACCESS_KEY))
  const [refreshToken, setRefresh] = useState(() => localStorage.getItem(REFRESH_KEY))
  const [user, setUser] = useState(null)
  const [loading, setLoading] = useState(true)

  const persistTokens = useCallback((access, refresh) => {
    setToken(access)
    setRefresh(refresh)
    setAccessToken(access)
    localStorage.setItem(ACCESS_KEY, access)
    localStorage.setItem(REFRESH_KEY, refresh)
  }, [])

  const clearAuth = useCallback(() => {
    setToken(null)
    setRefresh(null)
    setUser(null)
    setAccessToken(null)
    localStorage.removeItem(ACCESS_KEY)
    localStorage.removeItem(REFRESH_KEY)
  }, [])

  const login = useCallback(
    async (email, password) => {
      const { data } = await api.post('/api/auth/login', { email, password })
      persistTokens(data.access_token, data.refresh_token)
      const me = await api.get('/api/auth/me')
      setUser(me.data)
      return me.data
    },
    [persistTokens],
  )

  const register = useCallback(
    async (email, password) => {
      const { data } = await api.post('/api/auth/register', { email, password })
      persistTokens(data.access_token, data.refresh_token)
      const me = await api.get('/api/auth/me')
      setUser(me.data)
      return me.data
    },
    [persistTokens],
  )

  const logout = useCallback(() => {
    clearAuth()
  }, [clearAuth])

  useEffect(() => {
    const bootstrap = async () => {
      if (!accessToken) {
        setLoading(false)
        return
      }
      setAccessToken(accessToken)
      try {
        const { data } = await api.get('/api/auth/me')
        setUser(data)
      } catch {
        if (refreshToken) {
          try {
            const { data } = await api.post('/api/auth/refresh', {
              refresh_token: refreshToken,
            })
            persistTokens(data.access_token, data.refresh_token)
            const me = await api.get('/api/auth/me')
            setUser(me.data)
          } catch {
            clearAuth()
          }
        } else {
          clearAuth()
        }
      } finally {
        setLoading(false)
      }
    }
    bootstrap()
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const value = useMemo(
    () => ({
      user,
      accessToken,
      loading,
      isAuthenticated: Boolean(user && accessToken),
      login,
      register,
      logout,
    }),
    [user, accessToken, loading, login, register, logout],
  )

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth() {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used within AuthProvider')
  return ctx
}
