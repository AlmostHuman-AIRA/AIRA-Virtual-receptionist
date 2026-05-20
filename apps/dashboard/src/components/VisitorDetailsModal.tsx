import { X, Mail, Phone, Calendar, Eye } from 'lucide-react'
import { getVisitorPhotoUrl } from '../api'
import type { Visitor, LogEntry } from '../api'
import { useState } from 'react'

interface VisitorDetailsModalProps {
  visitor: Visitor;
  logs: LogEntry[];
  onClose: () => void;
}

function formatDate(dt: string | null): string {
  if (!dt) return '—';
  const d = new Date(dt + 'Z');
  return d.toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

export default function VisitorDetailsModal({ visitor, logs, onClose }: VisitorDetailsModalProps) {
  const [imgError, setImgError] = useState(false);
  const visitorLogs = logs.filter((l) => l.visitor_id === visitor.id);
  const initials = visitor.name
    .split(' ')
    .map((w) => w[0])
    .join('')
    .toUpperCase()
    .slice(0, 2);

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal__header">
          <h2 className="modal__title">Visitor Profile</h2>
          <button className="modal__close" onClick={onClose}>
            <X size={16} />
          </button>
        </div>
        <div className="modal__body">
          {/* Hero section */}
          <div className="modal__visitor-hero">
            {visitor.has_photo && !imgError ? (
              <img
                className="modal__visitor-photo"
                src={getVisitorPhotoUrl(visitor.id)}
                alt={visitor.name}
                onError={() => setImgError(true)}
              />
            ) : (
              <div className="modal__visitor-photo-placeholder">{initials}</div>
            )}
            <div className="modal__visitor-info">
              <h2>{visitor.name}</h2>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 6 }}>
                <Eye size={14} style={{ color: 'var(--text-muted)' }} />
                <span style={{ color: 'var(--text-secondary)', fontSize: 13 }}>
                  {visitor.total_visits} visit{visitor.total_visits !== 1 ? 's' : ''}
                </span>
              </div>
            </div>
          </div>

          {/* Detail cards */}
          <div className="modal__detail-grid">
            <div className="modal__detail-item">
              <div className="modal__detail-label">
                <Mail size={11} style={{ marginRight: 4, verticalAlign: -1 }} />
                Email
              </div>
              <div className="modal__detail-value">{visitor.email || '—'}</div>
            </div>
            <div className="modal__detail-item">
              <div className="modal__detail-label">
                <Phone size={11} style={{ marginRight: 4, verticalAlign: -1 }} />
                Phone
              </div>
              <div className="modal__detail-value">{visitor.phone || '—'}</div>
            </div>
            <div className="modal__detail-item">
              <div className="modal__detail-label">
                <Calendar size={11} style={{ marginRight: 4, verticalAlign: -1 }} />
                First Seen
              </div>
              <div className="modal__detail-value">{formatDate(visitor.first_seen)}</div>
            </div>
            <div className="modal__detail-item">
              <div className="modal__detail-label">
                <Calendar size={11} style={{ marginRight: 4, verticalAlign: -1 }} />
                Last Seen
              </div>
              <div className="modal__detail-value">{formatDate(visitor.last_seen)}</div>
            </div>
          </div>

          {/* Visit history */}
          {visitorLogs.length > 0 && (
            <>
              <h3 className="modal__section-title">Visit History</h3>
              <div className="modal__visit-list">
                {visitorLogs.slice(0, 10).map((log) => (
                  <div key={log.id} className="modal__visit-item">
                    <div>
                      <span style={{ fontWeight: 500 }}>{formatDate(log.check_in_time)}</span>
                      {log.badge_id && (
                        <span
                          style={{
                            marginLeft: 8,
                            fontSize: 11,
                            color: 'var(--text-muted)',
                            fontFamily: 'monospace',
                          }}
                        >
                          {log.badge_id}
                        </span>
                      )}
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      {log.purpose && (
                        <span className="modal__visit-item__purpose" title={log.purpose}>
                          {log.purpose}
                        </span>
                      )}
                      {log.is_active ? (
                        <span className="badge badge--success">
                          <span className="badge__dot" />
                          Active
                        </span>
                      ) : (
                        <span className="badge badge--neutral">Completed</span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
