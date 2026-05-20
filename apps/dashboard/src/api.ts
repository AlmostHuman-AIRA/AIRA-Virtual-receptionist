const API_BASE = '/api/dashboard';

export interface Stats {
  total_visitors: number;
  active_visits: number;
  todays_meetings: number;
  pending_checkouts: number;
  recent_checkins_today: number;
}

export interface Visitor {
  id: number;
  name: string;
  email: string | null;
  phone: string | null;
  has_photo: boolean;
  first_seen: string | null;
  last_seen: string | null;
  total_visits: number;
}

export interface Meeting {
  id: number;
  host_name: string;
  host_department: string | null;
  visitor_name: string | null;
  visitor_id: number | null;
  scheduled_start: string;
  scheduled_end: string | null;
  purpose: string | null;
  status: string;
  created_at: string | null;
}

export interface LogEntry {
  id: number;
  visitor_name: string | null;
  visitor_id: number | null;
  employee_name: string | null;
  person_type: string;
  badge_id: string | null;
  check_in_time: string | null;
  check_out_time: string | null;
  purpose: string | null;
  is_active: boolean;
}

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`API Error ${res.status}: ${err}`);
  }
  return res.json();
}

export async function fetchStats(): Promise<Stats> {
  const res = await fetch(`${API_BASE}/stats`);
  return handleResponse<Stats>(res);
}

export async function fetchVisitors(search?: string): Promise<Visitor[]> {
  const params = search ? `?search=${encodeURIComponent(search)}` : '';
  const res = await fetch(`${API_BASE}/visitors${params}`);
  return handleResponse<Visitor[]>(res);
}

export async function fetchMeetings(search?: string, status?: string): Promise<Meeting[]> {
  const params = new URLSearchParams();
  if (search) params.set('search', search);
  if (status) params.set('status', status);
  const qs = params.toString() ? `?${params.toString()}` : '';
  const res = await fetch(`${API_BASE}/meetings${qs}`);
  return handleResponse<Meeting[]>(res);
}

export async function fetchLogs(search?: string, activeOnly?: boolean): Promise<LogEntry[]> {
  const params = new URLSearchParams();
  if (search) params.set('search', search);
  if (activeOnly) params.set('active_only', 'true');
  const qs = params.toString() ? `?${params.toString()}` : '';
  const res = await fetch(`${API_BASE}/logs${qs}`);
  return handleResponse<LogEntry[]>(res);
}

export async function checkoutLog(logId: number): Promise<{ success: boolean }> {
  const res = await fetch(`${API_BASE}/logs/${logId}/checkout`, { method: 'POST' });
  return handleResponse<{ success: boolean }>(res);
}

export function getVisitorPhotoUrl(visitorId: number): string {
  return `${API_BASE}/visitors/${visitorId}/photo`;
}
