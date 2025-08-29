import { NavLink, Route, Routes } from 'react-router-dom'
import LiveMonitor from './pages/LiveMonitor'
import Ingest from './pages/Ingest'
import Explorer from './pages/Explorer'
import Guardrails from './pages/Guardrails'
import Settings from './pages/Settings'

export default function App(){
  return (
    <div style={{minHeight:'100%'}}>
      <nav className="navbar">
        <div className="brand">Bioelectric Control Panel</div>
        <NavLink to="/" end>Live</NavLink>
        <NavLink to="/ingest">Ingest</NavLink>
        <NavLink to="/explorer">Explorer</NavLink>
        <NavLink to="/guardrails">Guardrails</NavLink>
        <NavLink to="/settings">Settings</NavLink>
      </nav>
      <div className="container">
        <Routes>
          <Route path="/" element={<LiveMonitor/>}/>
          <Route path="/ingest" element={<Ingest/>}/>
          <Route path="/explorer" element={<Explorer/>}/>
          <Route path="/guardrails" element={<Guardrails/>}/>
          <Route path="/settings" element={<Settings/>}/>
        </Routes>
      </div>
    </div>
  )
}

