-- OPTIONAL frontend persistence handoff. Not applied by backend startup.
-- Review in Supabase staging with an owner role; assumes existing analysis_results.
-- Does not change RLS, grants or the analysis_results_safe view.
BEGIN;
ALTER TABLE public.analysis_results
  ADD COLUMN IF NOT EXISTS regional_analysis jsonb;
COMMENT ON COLUMN public.analysis_results.regional_analysis IS
  'Optional CAE v3 regional envelope. Preserve schema/scoring/catalog versions and coverage. NULL means not assessed regionally.';
COMMIT;

-- Also update the existing safe view's explicit projection and regenerate client
-- types in the frontend repo. Preserve all existing privacy exclusions/RLS.
-- Rollback: retain this nullable column to avoid losing historical regional data.
