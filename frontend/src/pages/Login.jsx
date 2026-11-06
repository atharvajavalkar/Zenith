import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { AuthField, AuthForm, AuthLayout } from '../components/AuthForm'
import { useAuth } from '../contexts/AuthContext'

export default function Login() {
  const { login } = useAuth()
  const navigate = useNavigate()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [submitting, setSubmitting] = useState(false)

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    setSubmitting(true)
    try {
      await login(email, password)
      navigate('/dashboard')
    } catch (err) {
      setError(err.response?.data?.detail || 'Login failed')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <AuthLayout title="Sign in" subtitle="Private access to the gold desk">
      <AuthForm
        onSubmit={handleSubmit}
        submitLabel={submitting ? 'Signing in…' : 'Enter Zenith'}
        error={error}
        footer={
          <p className="text-center text-sm font-light text-surface-muted mt-5">
            No account?{' '}
            <Link to="/register" className="text-accent-gold hover:text-accent-gold-soft transition">
              Register
            </Link>
          </p>
        }
      >
        <AuthField
          label="Email"
          type="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          placeholder="you@example.com"
        />
        <AuthField
          label="Password"
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          placeholder="••••••••"
        />
      </AuthForm>
    </AuthLayout>
  )
}
