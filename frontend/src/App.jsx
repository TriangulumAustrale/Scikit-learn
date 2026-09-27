import { useState } from 'react'
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  CircularProgress,
  Container,
  Divider,
  Stack,
  TextField,
  Typography,
} from '@mui/material'

const API_URL = import.meta.env.VITE_API_URL

// Plain-English labels and starting values for the model's 8 raw features,
// so the form is usable without reading the dataset docs.
const FIELDS = [
  { name: 'MedInc', label: 'Median Income (10k USD)', helperText: 'e.g. 8.3 = $83,000', default: '8.3' },
  { name: 'HouseAge', label: 'Median House Age (years)', helperText: '1 to 52', default: '25' },
  { name: 'AveRooms', label: 'Average Rooms per Household', helperText: 'Includes all room types', default: '6.2' },
  { name: 'AveBedrms', label: 'Average Bedrooms per Household', helperText: 'Usually near 1', default: '1.1' },
  { name: 'Population', label: 'Neighbourhood Population', helperText: 'People in the census block', default: '1500' },
  { name: 'AveOccup', label: 'Average People per Household', helperText: 'Household size', default: '3.0' },
  { name: 'Latitude', label: 'Latitude', helperText: '32 to 42 (California)', default: '34.05' },
  { name: 'Longitude', label: 'Longitude', helperText: '-125 to -114 (California)', default: '-118.25' },
]

// How each feature reads in the result card's explanation.
const FEATURE_PHRASES = {
  MedInc: 'Median income in this area',
  HouseAge: 'The age of homes in this area',
  AveRooms: 'The average number of rooms per home',
  AveBedrms: 'The average number of bedrooms per home',
  Population: 'The size of the local population',
  AveOccup: 'The average number of people per household',
  Latitude: 'The north-south location',
  Longitude: 'The east-west location',
}

const RANKS = ['the biggest driver', 'the second-largest driver', 'the third-largest driver']

const DEFAULTS = Object.fromEntries(FIELDS.map((f) => [f.name, f.default]))

const usd = new Intl.NumberFormat('en-US', {
  style: 'currency',
  currency: 'USD',
  maximumFractionDigits: 0,
})

function errorMessage(response, body) {
  // FastAPI returns 422 with a detail array naming the offending fields.
  if (Array.isArray(body?.detail)) {
    return body.detail
      .map((d) => {
        const field = d.loc?.[d.loc.length - 1]
        const label = FIELDS.find((f) => f.name === field)?.label ?? field
        return `${label}: ${d.msg}`
      })
      .join('\n')
  }
  return `The server returned ${response.status}. Please try again.`
}

export default function App() {
  const [values, setValues] = useState(DEFAULTS)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)

  const handleChange = (name) => (event) => {
    setValues((prev) => ({ ...prev, [name]: event.target.value }))
  }

  const handleSubmit = async (event) => {
    event.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)

    // Blank stays null rather than becoming 0, so the API reports it as missing.
    const payload = Object.fromEntries(
      FIELDS.map((f) => [f.name, values[f.name] === '' ? null : Number(values[f.name])]),
    )

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      const body = await response.json().catch(() => null)
      if (!response.ok) {
        setError(errorMessage(response, body))
        return
      }
      setResult(body)
    } catch {
      setError(`Could not reach the prediction service at ${API_URL}. Is the backend running?`)
    } finally {
      setLoading(false)
    }
  }

  return (
    <Container maxWidth="md" sx={{ py: 5 }}>
      <Typography variant="h4" component="h1" gutterBottom>
        House Price Predictor
      </Typography>
      <Typography color="text.secondary" sx={{ mb: 3 }}>
        Estimate the median house value for a California neighbourhood, and see which
        details mattered most.
      </Typography>

      <Card variant="outlined">
        <CardContent>
          <Box component="form" onSubmit={handleSubmit} noValidate>
            <Box
              sx={{
                display: 'grid',
                gap: 2,
                gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr' },
              }}
            >
              {FIELDS.map((field) => (
                <TextField
                  key={field.name}
                  label={field.label}
                  helperText={field.helperText}
                  value={values[field.name]}
                  onChange={handleChange(field.name)}
                  type="number"
                  slotProps={{ htmlInput: { step: 'any' } }}
                  fullWidth
                />
              ))}
            </Box>

            <Stack direction="row" spacing={2} sx={{ mt: 3, alignItems: 'center' }}>
              <Button type="submit" variant="contained" size="large" disabled={loading}>
                Predict price
              </Button>
              <Button onClick={() => setValues(DEFAULTS)} disabled={loading}>
                Reset
              </Button>
              {loading && <CircularProgress size={24} />}
            </Stack>
          </Box>
        </CardContent>
      </Card>

      {error && (
        <Alert severity="error" sx={{ mt: 3, whiteSpace: 'pre-line' }}>
          {error}
        </Alert>
      )}

      {result && (
        <Card variant="outlined" sx={{ mt: 3 }}>
          <CardContent>
            <Typography color="text.secondary" gutterBottom>
              Estimated median house value
            </Typography>
            <Typography variant="h3" component="p" sx={{ fontWeight: 600 }}>
              {usd.format(result.predicted_value_usd)}
            </Typography>

            <Divider sx={{ my: 2 }} />

            <Typography variant="subtitle1" gutterBottom>
              What drove this estimate
            </Typography>
            <Stack component="ul" spacing={1} sx={{ pl: 3, m: 0 }}>
              {result.top_features.map((item, index) => (
                <Typography component="li" key={item.feature}>
                  {FEATURE_PHRASES[item.feature] ?? item.feature} was{' '}
                  {RANKS[index] ?? 'a contributing factor'} of this estimate.
                </Typography>
              ))}
            </Stack>
          </CardContent>
        </Card>
      )}
    </Container>
  )
}
