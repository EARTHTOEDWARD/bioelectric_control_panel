const TOKEN_KEY = 'bcp_auth_token'

export function setToken(tok: string){
  localStorage.setItem(TOKEN_KEY, tok || '')
}
export function getToken(): string | null {
  const t = localStorage.getItem(TOKEN_KEY)
  return t && t.length ? t : null
}
export function authHeaders(): HeadersInit {
  const t = getToken()
  return t ? { 'Authorization': `Bearer ${t}` } : {}
}

