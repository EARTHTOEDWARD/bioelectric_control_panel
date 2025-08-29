import { useState } from 'react'
import { authHeaders } from '../utils/auth'

export default function Ingest(){
  const [file, setFile] = useState<File|null>(null)
  const [log, setLog] = useState<string>('')
  const apiUrl = (import.meta.env.VITE_API_URL as string) || 'http://127.0.0.1:8000'

  async function upload(){
    if (!file){ alert('Choose an NWB file first'); return }
    const form = new FormData()
    form.append('file', file)
    try{
      const res = await fetch(`${apiUrl}/v1/ingest/nwb-upload`, { method:'POST', body: form, headers: { ...authHeaders() } })
      const txt = await res.text()
      setLog(`Status ${res.status}\n` + txt)
    }catch(e:any){
      setLog('Upload failed: '+ (e?.message || String(e)))
    }
  }

  return <div className="grid" style={{gridTemplateColumns:'repeat(12,1fr)'}}>
    <div className="card" style={{gridColumn:'span 8'}}>
      <h3>Upload NWB</h3>
      <div className="controls">
        <input className="input" type="file" accept=".nwb,.h5" onChange={e=> setFile(e.target.files?.[0] || null) }/>
        <button className="button" onClick={upload}>Upload</button>
      </div>
      <div style={{height:8}}/>
      <div className="kv"><span className="k">API:</span><span>{apiUrl}</span></div>
    </div>
    <div className="card" style={{gridColumn:'span 4'}}>
      <h3>Logs</h3>
      <pre className="subtle" style={{whiteSpace:'pre-wrap'}}>{log || '—'}</pre>
    </div>
  </div>
}
