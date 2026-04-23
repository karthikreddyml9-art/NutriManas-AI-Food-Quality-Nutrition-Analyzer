-- NutriManas Supabase Schema
-- Run this in your Supabase SQL editor

-- ─────────────────────────────────────────────
-- User health profiles
-- ─────────────────────────────────────────────
create table if not exists profiles (
  id uuid primary key references auth.users(id) on delete cascade,
  age int default 25,
  gender text default 'male',
  weight_kg float default 70,
  height_cm float default 170,
  bmi float,
  activity_level text default 'moderate',
  health_goal text default 'general wellness',
  health_conditions text[] default '{}',
  updated_at timestamptz default now()
);

-- RLS: users can only read/write their own profile
alter table profiles enable row level security;

create policy "Users can view own profile"
  on profiles for select using (auth.uid() = id);

create policy "Users can insert own profile"
  on profiles for insert with check (auth.uid() = id);

create policy "Users can update own profile"
  on profiles for update using (auth.uid() = id);

-- ─────────────────────────────────────────────
-- Scan history
-- ─────────────────────────────────────────────
create table if not exists scans (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references auth.users(id) on delete cascade,
  food_name text not null,
  result jsonb not null,
  created_at timestamptz default now()
);

-- Index for fast per-user queries
create index scans_user_id_idx on scans(user_id);
create index scans_created_at_idx on scans(created_at desc);

-- RLS: users can only access their own scans
alter table scans enable row level security;

create policy "Users can view own scans"
  on scans for select using (auth.uid() = user_id);

create policy "Users can insert own scans"
  on scans for insert with check (auth.uid() = user_id);

create policy "Users can delete own scans"
  on scans for delete using (auth.uid() = user_id);

create policy "Users can update own scans"
  on scans for update using (auth.uid() = user_id);

-- Add feedback column (run this if table already exists)
alter table scans add column if not exists feedback text check (feedback in ('accurate', 'inaccurate'));
alter table scans add column if not exists feedback_note text;
