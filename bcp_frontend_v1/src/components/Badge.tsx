export default function Badge({ text, level='ok' }:{ text: string, level?: 'ok'|'warn'|'err' }){
  return <span className={`badge ${level}`}>{text}</span>
}

