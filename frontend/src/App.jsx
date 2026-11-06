import { Navigate, Route, Routes } from 'react-router-dom'
import ProtectedRoute from './components/ProtectedRoute'
import BacktestPage from './pages/Backtest'
import ChartPage from './pages/Chart'
import Dashboard from './pages/Dashboard'
import Login from './pages/Login'
import Register from './pages/Register'
import SettingsPage from './pages/Settings'
import NebulaPage from './pages/Nebula'
import StrategyLabPage from './pages/StrategyLab'
import WorldPage from './pages/World'

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route path="/register" element={<Register />} />
      <Route
        path="/dashboard"
        element={
          <ProtectedRoute>
            <Dashboard />
          </ProtectedRoute>
        }
      />
      <Route
        path="/world"
        element={
          <ProtectedRoute>
            <WorldPage />
          </ProtectedRoute>
        }
      />
      <Route
        path="/chart"
        element={
          <ProtectedRoute>
            <ChartPage />
          </ProtectedRoute>
        }
      />
      <Route
        path="/backtest"
        element={
          <ProtectedRoute>
            <BacktestPage />
          </ProtectedRoute>
        }
      />
      <Route
        path="/settings"
        element={
          <ProtectedRoute>
            <SettingsPage />
          </ProtectedRoute>
        }
      />
      <Route
        path="/strategy-lab"
        element={
          <ProtectedRoute>
            <StrategyLabPage />
          </ProtectedRoute>
        }
      />
      <Route
        path="/nebula"
        element={
          <ProtectedRoute>
            <NebulaPage />
          </ProtectedRoute>
        }
      />
      <Route path="*" element={<Navigate to="/dashboard" replace />} />
    </Routes>
  )
}
