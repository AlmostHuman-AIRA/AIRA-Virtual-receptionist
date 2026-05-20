import { Users, Activity, Calendar, Clock } from 'lucide-react'
import type { Stats } from '../api'

interface StatsGridProps {
  stats: Stats | null;
  loading: boolean;
}

export default function StatsGrid({ stats, loading }: StatsGridProps) {
  if (loading || !stats) {
    return (
      <div className="stats-grid">
        {[1, 2, 3, 4].map((i) => (
          <div key={i} className="stat-card" style={{ opacity: 0.5 }}>
            <div className="stat-card__header">
              <span className="stat-card__label">Loading...</span>
            </div>
            <div className="stat-card__value" style={{ color: 'var(--text-muted)' }}>—</div>
          </div>
        ))}
      </div>
    );
  }

  const cards = [
    {
      label: 'Registered Visitors',
      value: stats.total_visitors,
      sub: 'All-time profiles',
      variant: 'visitors' as const,
      icon: <Users size={18} />,
    },
    {
      label: 'Active Visits',
      value: stats.active_visits,
      sub: 'Currently in building',
      variant: 'active' as const,
      icon: <Activity size={18} />,
    },
    {
      label: "Today's Meetings",
      value: stats.todays_meetings,
      sub: 'Scheduled for today',
      variant: 'meetings' as const,
      icon: <Calendar size={18} />,
    },
    {
      label: 'Pending Checkouts',
      value: stats.pending_checkouts,
      sub: `${stats.recent_checkins_today} check-ins today`,
      variant: 'pending' as const,
      icon: <Clock size={18} />,
    },
  ];

  return (
    <div className="stats-grid">
      {cards.map((card) => (
        <div key={card.variant} className={`stat-card stat-card--${card.variant}`}>
          <div className="stat-card__header">
            <span className="stat-card__label">{card.label}</span>
            <div className="stat-card__icon">{card.icon}</div>
          </div>
          <div className="stat-card__value">{card.value}</div>
          <div className="stat-card__sub">{card.sub}</div>
        </div>
      ))}
    </div>
  );
}
