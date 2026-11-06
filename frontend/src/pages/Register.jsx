import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { AuthField, AuthForm, AuthLayout } from '../components/AuthForm'
import { useAuth } from '../contexts/AuthContext'

export default function Register() {
  const { register } = useAuth()
  const navigate = useNavigate()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [confirm, setConfirm] = useState('')
  const [error, setError] = useState('')
  const [submitting, setSubmitting] = useState(false)

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    if (password !== confirm) {
      setError('Passwords do not match')
      return
    }
    setSubmitting(true)
    try {
      await register(email, password)
      navigate('/dashboard')
    } catch (err) {
      const detail = err.response?.data?.detail
      setError(Array.isArray(detail) ? detail[0]?.msg : detail || 'Registration failed')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <AuthLayout title="Create account" subtitle="Request access to the private desk">
      <AuthForm
        onSubmit={handleSubmit}
        submitLabel={submitting ? 'Creating…' : 'Create account'}
        error={error}
        footer={
          <p className="text-center text-sm font-light text-surface-muted mt-5">
            Already have an account?{' '}
            <Link to="/login" className="text-accent-gold hover:text-accent-gold-soft transition">
              Sign in
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
          placeholder="Min 6 characters"
        />
        <AuthField
          label="Confirm password"
          type="password"
          value={confirm}
          onChange={(e) => setConfirm(e.target.value)}
          placeholder="Repeat password"
        />
      </AuthForm>
    </AuthLayout>
  )
}
