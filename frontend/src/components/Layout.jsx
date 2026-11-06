import { Link, useLocation } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import ZenithLogo from './ZenithLogo'
import TradeModeSelect from './TradeModeSelect'

export default function Layout({ children, variant = 'default' }) {
  const { user, logout } = useAuth()
  const location = useLocation()
  const isChart = variant === 'chart'

  const navLink = (to, label) => {
    const active = location.pathname === to
    return (
      <Link to={to} className={`nav-link ${active ? 'nav-link-active' : ''}`}>
        {label}
      </Link>
    )
  }

  return (
    <div className="app-shell">
      <header className="relative z-40 sticky top-0 border-b border-accent-gold/10 bg-[#080807]/80 backdrop-blur-xl shrink-0">
        <div
          className={`${
            isChart ? 'w-full' : 'max-w-7xl mx-auto'
          } px-5 sm:px-8 h-[4.25rem] flex items-center justify-between gap-6`}
        >
          <div className="flex items-center gap-8 min-w-0">
            <Link to="/dashboard" className="shrink-0 group">
              <ZenithLogo
                size={26}
                variant="dark"
                showTagline={false}
                textClassName="text-[1.45rem] sm:text-[1.6rem] transition-opacity group-hover:opacity-90"
              />
            </Link>
            <nav className="hidden sm:flex items-center gap-1 overflow-x-auto">
              {navLink('/dashboard', 'Dashboard')}
              {navLink('/world', 'World')}
              {navLink('/chart', 'Chart')}
              {navLink('/backtest', 'Backtest')}
              {navLink('/strategy-lab', 'Strategy Lab')}
              {navLink('/nebula', 'Nebula')}
              {navLink('/settings', 'Settings')}
            </nav>
          </div>
          <div className="flex items-center gap-3 sm:gap-4 shrink-0">
            <TradeModeSelect compact />
            {user?.email && (
              <span
                className="hidden md:block text-[11px] tracking-wide font-light truncate max-w-[160px] text-surface-muted"
                title={user.email}
              >
                {user.email}
              </span>
            )}
            <button type="button" onClick={logout} className="btn-secondary">
              Logout
            </button>
          </div>
        </div>
        <nav className="sm:hidden flex items-center gap-0.5 overflow-x-auto px-4 pb-3 border-t border-accent-gold/5">
          {navLink('/dashboard', 'Dashboard')}
          {navLink('/world', 'World')}
          {navLink('/chart', 'Chart')}
          {navLink('/backtest', 'Backtest')}
          {navLink('/strategy-lab', 'Lab')}
          {navLink('/nebula', 'Nebula')}
          {navLink('/settings', 'Settings')}
        </nav>
      </header>
      <main
        className={
          isChart
            ? 'relative z-[1]'
            : 'relative z-[1] max-w-7xl mx-auto px-5 sm:px-10 py-10 sm:py-12 space-y-10 fade-page'
        }
      >
        {children}
      </main>
    </div>
  )
}
