import { Link, useLocation } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import ZenithLogo from './ZenithLogo'

export default function Layout({ children, variant = 'default' }) {
  const { user, logout } = useAuth()
  const location = useLocation()
  const isChart = variant === 'chart'

  const navLink = (to, label) => {
    const active = location.pathname === to
    if (isChart) {
      return (
        <Link
          to={to}
          className="text-sm px-3 py-1.5 rounded-lg transition"
          style={active ? { backgroundColor: '#F0F3FA', color: '#131722' } : { color: '#787B86' }}
        >
          {label}
        </Link>
      )
    }
    return (
      <Link
        to={to}
        className={`relative text-sm font-medium px-3.5 py-2 rounded-lg transition-colors ${
          active
            ? 'text-surface bg-gold-gradient shadow-glow'
            : 'text-surface-muted hover:text-gray-100 hover:bg-white/5'
        }`}
      >
        {label}
      </Link>
    )
  }

  return (
    <div className={`min-h-screen ${isChart ? 'bg-white' : 'bg-surface'}`}>
      <header
        className={`shrink-0 ${
          isChart
            ? 'border-b'
            : 'sticky top-0 z-40 border-b border-surface-border/70 bg-surface/80 backdrop-blur-xl'
        }`}
        style={{
          backgroundColor: isChart ? '#FFFFFF' : undefined,
          borderColor: isChart ? '#E0E3EB' : undefined,
        }}
      >
        <div
          className={`${isChart ? 'w-full' : 'max-w-7xl mx-auto'} px-4 py-3 flex items-center justify-between`}
        >
          <div className="flex items-center gap-7">
            <Link to="/dashboard" className="hover:opacity-95 transition-opacity">
              <ZenithLogo
                size={44}
                variant={isChart ? 'light' : 'dark'}
                showTagline={!isChart}
                textClassName="text-3xl sm:text-4xl"
              />
            </Link>
            <nav className="flex items-center gap-1.5">
              {navLink('/dashboard', 'Dashboard')}
              {navLink('/chart', 'Chart')}
            </nav>
          </div>
          <div className="flex items-center gap-3">
            {user?.email && (
              <div className="hidden sm:flex items-center gap-2.5 pr-1">
                <span
                  className="grid h-8 w-8 place-items-center rounded-full bg-gold-gradient text-sm font-bold text-surface"
                  style={isChart ? {} : undefined}
                  title={user.email}
                >
                  {user.email[0]?.toUpperCase()}
                </span>
                <span
                  className="text-sm"
                  style={{ color: isChart ? '#787B86' : undefined }}
                >
                  {user.email}
                </span>
              </div>
            )}
            <button
              type="button"
              onClick={logout}
              className={`text-sm px-4 py-2 rounded-lg border transition ${
                isChart ? '' : 'border-surface-border text-surface-muted hover:text-gray-100 hover:border-accent-gold/40'
              }`}
              style={isChart ? { borderColor: '#E0E3EB', color: '#131722' } : undefined}
            >
              Logout
            </button>
          </div>
        </div>
      </header>
      <main className={isChart ? '' : 'max-w-7xl mx-auto px-4 py-7 space-y-7'}>{children}</main>
    </div>
  )
}
