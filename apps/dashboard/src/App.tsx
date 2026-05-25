import { useState, useEffect, useCallback } from 'react';
import {
  Search,
  Users,
  Calendar,
  ClipboardList,
  RefreshCw,
  LogOut,
  UserCircle,
  Inbox,
} from 'lucide-react';
import {
  fetchStats,
  fetchVisitors,
  fetchMeetings,
  fetchLogs,
  checkoutLog,
  getVisitorPhotoUrl,
} from './api';
import type { Stats, Visitor, Meeting, LogEntry } from './api';
import StatsGrid from './components/StatsGrid';
import VisitorDetailsModal from './components/VisitorDetailsModal';

type Tab = 'logs' | 'visitors' | 'meetings';

function formatDate(dt: string | null): string {
  if (!dt) return '—';
  const d = new Date(dt + 'Z');
  return d.toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

function formatDateFull(dt: string | null): string {
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

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>('logs');
  const [search, setSearch] = useState('');
  const [activeOnly, setActiveOnly] = useState(false);
  const [statusFilter, setStatusFilter] = useState('');

  const [stats, setStats] = useState<Stats | null>(null);
  const [visitors, setVisitors] = useState<Visitor[]>([]);
  const [meetings, setMeetings] = useState<Meeting[]>([]);
  const [logs, setLogs] = useState<LogEntry[]>([]);

  const [loading, setLoading] = useState(true);
  const [checkingOut, setCheckingOut] = useState<number | null>(null);
  const [selectedVisitor, setSelectedVisitor] = useState<Visitor | null>(null);
  const [lastRefresh, setLastRefresh] = useState<Date>(new Date());

  // ── Data fetching ─────────────────────────────────────────────────────────

  const loadData = useCallback(async () => {
    setLoading(true);
    try {
      const [s, v, m, l] = await Promise.all([
        fetchStats(),
        fetchVisitors(activeTab === 'visitors' ? search : undefined),
        fetchMeetings(
          activeTab === 'meetings' ? search : undefined,
          statusFilter || undefined
        ),
        fetchLogs(
          activeTab === 'logs' ? search : undefined,
          activeOnly
        ),
      ]);
      setStats(s);
      setVisitors(v);
      setMeetings(m);
      setLogs(l);
      setLastRefresh(new Date());
    } catch (err) {
      console.error('Failed to load dashboard data:', err);
    } finally {
      setLoading(false);
    }
  }, [activeTab, search, activeOnly, statusFilter]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  // Auto-refresh every 30 seconds
  useEffect(() => {
    const interval = setInterval(loadData, 30000);
    return () => clearInterval(interval);
  }, [loadData]);

  // ── Checkout handler ──────────────────────────────────────────────────────

  const handleCheckout = async (logId: number) => {
    setCheckingOut(logId);
    try {
      await checkoutLog(logId);
      await loadData();
    } catch (err) {
      console.error('Checkout failed:', err);
    } finally {
      setCheckingOut(null);
    }
  };

  // ── Status badge helper ───────────────────────────────────────────────────

  function statusBadge(status: string) {
    const map: Record<string, { cls: string; label: string }> = {
      scheduled: { cls: 'badge--info', label: 'Scheduled' },
      completed: { cls: 'badge--success', label: 'Completed' },
      cancelled: { cls: 'badge--danger', label: 'Cancelled' },
    };
    const s = map[status] || { cls: 'badge--neutral', label: status };
    return <span className={`badge ${s.cls}`}>{s.label}</span>;
  }

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div className="dashboard-layout">
      {/* Header */}
      <header className="dashboard-header">
        <div className="dashboard-header__brand">
          <div className="dashboard-header__logo">A</div>
          <div>
            <h1 className="dashboard-header__title">AIRA Dashboard</h1>
            <p className="dashboard-header__subtitle">Visitor Intelligence & Reception Analytics</p>
          </div>
        </div>
        <div className="dashboard-header__actions">
          <div className="refresh-indicator">
            <span className="refresh-indicator__dot" />
            <span>
              Updated {lastRefresh.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}
            </span>
          </div>
          <button className="btn btn--ghost" onClick={loadData} title="Refresh">
            <RefreshCw size={14} />
            Refresh
          </button>
        </div>
      </header>

      {/* Stats */}
      <StatsGrid stats={stats} loading={loading && !stats} />

      {/* Tabs */}
      <div className="tabs">
        <button
          className={`tab-btn ${activeTab === 'logs' ? 'tab-btn--active' : ''}`}
          onClick={() => { setActiveTab('logs'); setSearch(''); }}
        >
          <ClipboardList size={15} />
          Live Logs
          {stats && stats.active_visits > 0 && (
            <span className="tab-btn__badge">{stats.active_visits}</span>
          )}
        </button>
        <button
          className={`tab-btn ${activeTab === 'visitors' ? 'tab-btn--active' : ''}`}
          onClick={() => { setActiveTab('visitors'); setSearch(''); }}
        >
          <Users size={15} />
          Visitors
          {stats && <span className="tab-btn__badge">{stats.total_visitors}</span>}
        </button>
        <button
          className={`tab-btn ${activeTab === 'meetings' ? 'tab-btn--active' : ''}`}
          onClick={() => { setActiveTab('meetings'); setSearch(''); }}
        >
          <Calendar size={15} />
          Meetings
        </button>
      </div>

      {/* Toolbar */}
      <div className="toolbar">
        <div className="search-box">
          <Search size={16} className="search-box__icon" />
          <input
            className="search-box__input"
            type="text"
            placeholder={
              activeTab === 'logs'
                ? 'Search by name or badge ID...'
                : activeTab === 'visitors'
                ? 'Search visitors by name or email...'
                : 'Search meetings by host or visitor...'
            }
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>

        {activeTab === 'logs' && (
          <button
            className={`filter-btn ${activeOnly ? 'filter-btn--active' : ''}`}
            onClick={() => setActiveOnly(!activeOnly)}
          >
            <UserCircle size={14} />
            Active Only
          </button>
        )}

        {activeTab === 'meetings' && (
          <>
            {['', 'scheduled', 'completed', 'cancelled'].map((s) => (
              <button
                key={s}
                className={`filter-btn ${statusFilter === s ? 'filter-btn--active' : ''}`}
                onClick={() => setStatusFilter(s)}
              >
                {s || 'All'}
              </button>
            ))}
          </>
        )}
      </div>

      {/* Content */}
      {loading && !stats ? (
        <div className="spinner-container">
          <div className="spinner" />
        </div>
      ) : (
        <>
          {/* ──── LIVE LOGS TAB ────────────────────────────────────────────── */}
          {activeTab === 'logs' && (
            <div className="data-table-wrapper">
              {logs.length === 0 ? (
                <div className="empty-state">
                  <Inbox size={48} className="empty-state__icon" />
                  <h3 className="empty-state__title">No reception logs found</h3>
                  <p className="empty-state__desc">
                    Visitor check-ins will appear here in real-time as people arrive.
                  </p>
                </div>
              ) : (
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Visitor</th>
                      <th>Type</th>
                      <th>Badge ID</th>
                      <th>Meeting With</th>
                      <th>Check In</th>
                      <th>Check Out</th>
                      <th>Purpose</th>
                      <th>Status</th>
                      <th></th>
                    </tr>
                  </thead>
                  <tbody>
                    {logs.map((log) => {
                      const initials = (log.visitor_name || '?')
                        .split(' ')
                        .map((w) => w[0])
                        .join('')
                        .toUpperCase()
                        .slice(0, 2);
                      return (
                        <tr key={log.id}>
                          <td>
                            <div className="cell-avatar">
                              {log.visitor_id ? (
                                <img
                                  className="avatar"
                                  src={getVisitorPhotoUrl(log.visitor_id)}
                                  alt=""
                                  onError={(e) => {
                                    (e.target as HTMLImageElement).style.display = 'none';
                                    (e.target as HTMLImageElement).nextElementSibling?.classList.remove('hidden');
                                  }}
                                />
                              ) : null}
                              <div className="avatar-placeholder">{initials}</div>
                              <div>
                                <div className="cell-name">{log.visitor_name || 'Unknown'}</div>
                              </div>
                            </div>
                          </td>
                          <td>
                            <span className={`badge ${log.person_type === 'VISITOR' ? 'badge--info' : 'badge--neutral'}`}>
                              {log.person_type}
                            </span>
                          </td>
                          <td style={{ fontFamily: 'monospace', fontSize: 12, color: 'var(--text-muted)' }}>
                            {log.badge_id || '—'}
                          </td>
                          <td style={{ color: 'var(--text-secondary)' }}>
                            {log.employee_name || '—'}
                          </td>
                          <td style={{ fontSize: 13 }}>{formatDate(log.check_in_time)}</td>
                          <td style={{ fontSize: 13 }}>{formatDate(log.check_out_time)}</td>
                          <td
                            style={{
                              maxWidth: 180,
                              overflow: 'hidden',
                              textOverflow: 'ellipsis',
                              whiteSpace: 'nowrap',
                              color: 'var(--text-muted)',
                              fontSize: 13,
                            }}
                            title={log.purpose || ''}
                          >
                            {log.purpose || '—'}
                          </td>
                          <td>
                            {log.is_active ? (
                              <span className="badge badge--success">
                                <span className="badge__dot" />
                                Active
                              </span>
                            ) : (
                              <span className="badge badge--neutral">Done</span>
                            )}
                          </td>
                          <td>
                            {log.is_active && (
                              <button
                                className="btn btn--checkout"
                                disabled={checkingOut === log.id}
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleCheckout(log.id);
                                }}
                              >
                                <LogOut size={12} />
                                {checkingOut === log.id ? 'Checking out...' : 'Check Out'}
                              </button>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              )}
            </div>
          )}

          {/* ──── VISITORS TAB ─────────────────────────────────────────────── */}
          {activeTab === 'visitors' && (
            <div className="data-table-wrapper">
              {visitors.length === 0 ? (
                <div className="empty-state">
                  <Users size={48} className="empty-state__icon" />
                  <h3 className="empty-state__title">No visitors registered</h3>
                  <p className="empty-state__desc">
                    Visitor profiles are created automatically when someone checks in.
                  </p>
                </div>
              ) : (
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Visitor</th>
                      <th>Email</th>
                      <th>Phone</th>
                      <th>Total Visits</th>
                      <th>First Seen</th>
                      <th>Last Seen</th>
                    </tr>
                  </thead>
                  <tbody>
                    {visitors.map((v) => {
                      const initials = v.name
                        .split(' ')
                        .map((w) => w[0])
                        .join('')
                        .toUpperCase()
                        .slice(0, 2);
                      return (
                        <tr
                          key={v.id}
                          onClick={() => setSelectedVisitor(v)}
                        >
                          <td>
                            <div className="cell-avatar">
                              {v.has_photo ? (
                                <img
                                  className="avatar"
                                  src={getVisitorPhotoUrl(v.id)}
                                  alt={v.name}
                                  onError={(e) => {
                                    (e.target as HTMLImageElement).style.display = 'none';
                                  }}
                                />
                              ) : (
                                <div className="avatar-placeholder">{initials}</div>
                              )}
                              <div className="cell-name">{v.name}</div>
                            </div>
                          </td>
                          <td style={{ color: 'var(--text-secondary)', fontSize: 13 }}>
                            {v.email || '—'}
                          </td>
                          <td style={{ color: 'var(--text-secondary)', fontSize: 13 }}>
                            {v.phone || '—'}
                          </td>
                          <td>
                            <span className="badge badge--info">{v.total_visits}</span>
                          </td>
                          <td style={{ fontSize: 13 }}>{formatDateFull(v.first_seen)}</td>
                          <td style={{ fontSize: 13 }}>{formatDateFull(v.last_seen)}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              )}
            </div>
          )}

          {/* ──── MEETINGS TAB ─────────────────────────────────────────────── */}
          {activeTab === 'meetings' && (
            <div className="data-table-wrapper">
              {meetings.length === 0 ? (
                <div className="empty-state">
                  <Calendar size={48} className="empty-state__icon" />
                  <h3 className="empty-state__title">No meetings found</h3>
                  <p className="empty-state__desc">
                    Meetings are scheduled through the AIRA voice assistant.
                  </p>
                </div>
              ) : (
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Host</th>
                      <th>Department</th>
                      <th>Visitor</th>
                      <th>Scheduled</th>
                      <th>Purpose</th>
                      <th>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {meetings.map((m) => (
                      <tr key={m.id}>
                        <td>
                          <div className="cell-name">{m.host_name}</div>
                        </td>
                        <td style={{ color: 'var(--text-secondary)', fontSize: 13 }}>
                          {m.host_department || '—'}
                        </td>
                        <td>
                          <div className="cell-name" style={{ color: 'var(--text-accent)' }}>
                            {m.visitor_name || '—'}
                          </div>
                        </td>
                        <td style={{ fontSize: 13 }}>{formatDateFull(m.scheduled_start)}</td>
                        <td
                          style={{
                            maxWidth: 200,
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                            whiteSpace: 'nowrap',
                            color: 'var(--text-muted)',
                            fontSize: 13,
                          }}
                          title={m.purpose || ''}
                        >
                          {m.purpose || '—'}
                        </td>
                        <td>{statusBadge(m.status)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          )}
        </>
      )}

      {/* Visitor Details Modal */}
      {selectedVisitor && (
        <VisitorDetailsModal
          visitor={selectedVisitor}
          logs={logs}
          onClose={() => setSelectedVisitor(null)}
        />
      )}
    </div>
  );
}
