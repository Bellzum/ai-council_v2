"""
Hierarchy Validation Utilities for Medium Workflows

Provides utility functions for validating work item hierarchies, checking Feature
readiness, and identifying orphan/empty work items.

Design Pattern:
- All functions use the work tracking adapter (NO CLI commands)
- Return (is_valid: bool, issues: List[str]) or structured results
- External verification pattern from VISION.md
- Reusable across workflows (backlog grooming, sprint review, etc.)
"""

from typing import Dict, Any, List, Tuple, Optional


def validate_epic_hierarchy(adapter, epic_id: int) -> Tuple[bool, List[str]]:
    """
    Validate that an EPIC has child Features.

    Args:
        adapter: Work tracking adapter instance
        epic_id: EPIC work item ID

    Returns:
        Tuple of (is_valid: bool, issues: List[str])
        - is_valid: True if EPIC has at least one Feature child
        - issues: List of validation issues found

    Example:
        >>> is_valid, issues = validate_epic_hierarchy(adapter, 123)
        >>> if not is_valid:
        ...     print(f"EPIC 123 issues: {issues}")
    """
    issues = []

    try:
        # Query EPIC (external source of truth)
        epic = adapter.get_work_item(epic_id)

        if not epic:
            issues.append(f"EPIC {epic_id} not found in tracking system")
            return False, issues

        # Verify it's actually an EPIC
        work_item_type = epic.get('type')
        if not work_item_type:
            work_item_type = epic.get('fields', {}).get('System.WorkItemType')

        if work_item_type != 'Epic':
            issues.append(f"Work item {epic_id} is type '{work_item_type}', not EPIC")
            return False, issues

        # Check for child relationships
        relations = epic.get('relations', [])
        child_count = 0

        for relation in relations:
            rel_type = relation.get('rel', '')
            # Child relations are marked as "System.LinkTypes.Hierarchy-Forward"
            if 'Hierarchy-Forward' in rel_type or 'Child' in rel_type:
                child_count += 1

        if child_count == 0:
            issues.append(f"EPIC {epic_id} has no child Features")
            return False, issues

        # Valid EPIC with children
        return True, []

    except Exception as e:
        issues.append(f"Failed to validate EPIC {epic_id}: {str(e)}")
        return False, issues


def find_orphan_features(adapter) -> List[int]:
    """
    Find Features with no parent EPIC.

    Orphan Features are problematic because they lack strategic alignment
    and cannot be rolled up into EPIC-level tracking.

    Args:
        adapter: Work tracking adapter instance

    Returns:
        List of Feature IDs that have no parent EPIC

    Example:
        >>> orphan_ids = find_orphan_features(adapter)
        >>> if orphan_ids:
        ...     print(f"Found {len(orphan_ids)} orphan Features: {orphan_ids}")
    """
    orphan_ids = []

    try:
        # Query all Features (no iteration filter = backlog)
        features = adapter.query_work_items(work_item_type='Feature')

        for feature in features:
            feature_id = feature.get('id')
            if not feature_id:
                continue

            # Check for parent relationships
            relations = feature.get('relations', [])
            has_parent = False

            for relation in relations:
                rel_type = relation.get('rel', '')
                # Parent relations are marked as "System.LinkTypes.Hierarchy-Reverse"
                if 'Hierarchy-Reverse' in rel_type or 'Parent' in rel_type:
                    has_parent = True
                    break

            if not has_parent:
                orphan_ids.append(feature_id)

    except Exception as e:
        # Log error but don't crash - return partial results
        print(f"⚠️  Error finding orphan Features: {e}")

    return orphan_ids


def find_empty_epics(adapter) -> List[int]:
    """
    Find EPICs with no child Features.

    Empty EPICs indicate incomplete planning or EPICs that need to be
    broken down into Features.

    Args:
        adapter: Work tracking adapter instance

    Returns:
        List of EPIC IDs that have no child Features

    Example:
        >>> empty_ids = find_empty_epics(adapter)
        >>> if empty_ids:
        ...     print(f"Found {len(empty_ids)} empty EPICs: {empty_ids}")
    """
    empty_ids = []

    try:
        # Query all EPICs (no iteration filter = backlog)
        epics = adapter.query_work_items(work_item_type='Epic')

        for epic in epics:
            epic_id = epic.get('id')
            if not epic_id:
                continue

            # Check for child relationships
            relations = epic.get('relations', [])
            has_children = False

            for relation in relations:
                rel_type = relation.get('rel', '')
                # Child relations are marked as "System.LinkTypes.Hierarchy-Forward"
                if 'Hierarchy-Forward' in rel_type or 'Child' in rel_type:
                    has_children = True
                    break

            if not has_children:
                empty_ids.append(epic_id)

    except Exception as e:
        # Log error but don't crash - return partial results
        print(f"⚠️  Error finding empty EPICs: {e}")

    return empty_ids


def validate_feature_readiness(
    work_item: Dict[str, Any],
    min_description_length: int = 500,
    min_acceptance_criteria: int = 3
) -> Tuple[bool, List[str]]:
    """
    Check if Feature is ready for sprint (description, acceptance criteria, story points).

    A Feature is considered "ready" if it has:
    1. Description with sufficient detail (default: 500+ characters)
    2. Acceptance criteria (default: 3+ criteria)
    3. Story points estimated

    Args:
        work_item: Feature work item dict
        min_description_length: Minimum description length (default: 500)
        min_acceptance_criteria: Minimum acceptance criteria count (default: 3)

    Returns:
        Tuple of (is_ready: bool, missing: List[str])
        - is_ready: True if all readiness criteria met
        - missing: List of missing/incomplete criteria

    Example:
        >>> feature = adapter.get_work_item(456)
        >>> is_ready, missing = validate_feature_readiness(feature)
        >>> if not is_ready:
        ...     print(f"Feature not ready: {missing}")
    """
    missing = []

    try:
        # Get fields from work item
        fields = work_item.get('fields', {})

        # Check 1: Description length
        description = fields.get('System.Description', '')
        if not description:
            description = work_item.get('description', '')

        # Strip HTML tags for length check (Azure DevOps stores description as HTML)
        import re
        text_only = re.sub(r'<[^>]+>', '', description)
        text_only = text_only.strip()

        if len(text_only) < min_description_length:
            missing.append(
                f"Description too short ({len(text_only)} chars, need {min_description_length}+)"
            )

        # Check 2: Acceptance criteria
        # Azure DevOps stores acceptance criteria in System.AcceptanceCriteria field
        acceptance_criteria = fields.get('Microsoft.VSTS.Common.AcceptanceCriteria', '')
        if not acceptance_criteria:
            acceptance_criteria = work_item.get('acceptance_criteria', '')

        # Count criteria (look for bullet points, numbered lists, or newlines)
        criteria_count = 0
        if acceptance_criteria:
            # Strip HTML
            criteria_text = re.sub(r'<[^>]+>', '', acceptance_criteria)
            # Count lines starting with bullet/number or non-empty lines
            lines = [line.strip() for line in criteria_text.split('\n') if line.strip()]
            criteria_count = len(lines)

        if criteria_count < min_acceptance_criteria:
            missing.append(
                f"Not enough acceptance criteria ({criteria_count} found, need {min_acceptance_criteria}+)"
            )

        # Check 3: Story points estimated
        story_points = fields.get('Microsoft.VSTS.Scheduling.StoryPoints')
        if not story_points:
            story_points = work_item.get('story_points')

        if not story_points or story_points <= 0:
            missing.append("Story points not estimated")

        # Return readiness status
        is_ready = len(missing) == 0
        return is_ready, missing

    except Exception as e:
        missing.append(f"Failed to validate Feature readiness: {str(e)}")
        return False, missing


def check_epic_completion_eligibility(
    adapter,
    epic_id: int
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Check if EPIC can be auto-completed (all child Features Done).

    This implements the EPIC auto-completion logic for sprint review workflow.
    Uses external verification pattern: queries adapter for each child Feature
    to verify state (not trusting AI claims).

    Args:
        adapter: Work tracking adapter instance
        epic_id: EPIC work item ID

    Returns:
        Tuple of (eligible: bool, reason: str, evidence: Dict[str, Any])
        - eligible: True if EPIC can be transitioned to Done
        - reason: Human-readable reason for eligibility/ineligibility
        - evidence: Dict with verification details for audit

    Example:
        >>> eligible, reason, evidence = check_epic_completion_eligibility(adapter, 123)
        >>> if eligible:
        ...     adapter.update_work_item(123, state='Done')
        ...     print(f"EPIC 123 auto-completed: {reason}")
    """
    try:
        # 1. Query EPIC (external source of truth)
        epic = adapter.get_work_item(epic_id)
        if not epic:
            return (False, "EPIC not found", {})

        # 2. Get child Feature IDs from relations
        relations = epic.get('relations', [])
        child_ids = []

        for relation in relations:
            rel_type = relation.get('rel', '')
            if 'Hierarchy-Forward' in rel_type or 'Child' in rel_type:
                # Extract work item ID from URL
                url = relation.get('url', '')
                if url:
                    # URL format: https://dev.azure.com/org/project/_apis/wit/workItems/{id}
                    work_item_id = url.split('/')[-1]
                    try:
                        child_ids.append(int(work_item_id))
                    except (ValueError, TypeError):
                        continue

        if not child_ids:
            return (False, "EPIC has no child Features", {'epic_id': epic_id, 'child_ids': []})

        # 3. Query each child Feature to verify state (EXTERNAL VERIFICATION)
        child_states = {}
        for child_id in child_ids:
            child = adapter.get_work_item(child_id)
            if child and 'fields' in child:
                fields = child['fields']
                state = fields.get('System.State')
                child_type = fields.get('System.WorkItemType')

                # Only count Features (not Tasks or other children)
                if child_type == 'Feature':
                    child_states[child_id] = state

        if not child_states:
            return (False, "EPIC has no Feature children (may have other work item types)", {
                'epic_id': epic_id,
                'child_ids': child_ids,
                'feature_count': 0
            })

        # 4. Check if ALL Features are Done
        all_done = all(state == 'Done' for state in child_states.values())

        evidence = {
            'epic_id': epic_id,
            'child_feature_ids': list(child_states.keys()),
            'child_states': child_states,
            'all_features_done': all_done,
            'total_features': len(child_states),
            'done_features': sum(1 for state in child_states.values() if state == 'Done')
        }

        if all_done:
            return (True, "All child Features completed", evidence)
        else:
            incomplete = [cid for cid, state in child_states.items() if state != 'Done']
            return (False, f"{len(incomplete)} Features incomplete", evidence)

    except Exception as e:
        return (False, f"Error checking EPIC completion: {str(e)}", {})


def auto_complete_epic(
    adapter,
    epic_id: int,
    evidence: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Transition EPIC to Done with audit logging.

    This function implements rollback safety: if transition fails, it logs
    the error but doesn't crash the workflow. The workflow can continue
    and the user can manually complete the EPIC.

    Args:
        adapter: Work tracking adapter instance
        epic_id: EPIC work item ID
        evidence: Evidence dict from check_epic_completion_eligibility

    Returns:
        Dict with success status, epic_id, evidence, and error if failed

    Example:
        >>> eligible, reason, evidence = check_epic_completion_eligibility(adapter, 123)
        >>> if eligible:
        ...     result = auto_complete_epic(adapter, 123, evidence)
        ...     if result['success']:
        ...         print(f"EPIC 123 auto-completed")
    """
    try:
        # Transition EPIC to Done
        result = adapter.update_work_item(work_item_id=epic_id, state='Done', verify=True)

        # External verification - query to confirm state changed
        epic = adapter.get_work_item(epic_id)
        if epic and 'fields' in epic:
            new_state = epic['fields'].get('System.State')
            if new_state == 'Done':
                return {
                    'success': True,
                    'epic_id': epic_id,
                    'evidence': evidence,
                    'verified_state': new_state
                }

        # State transition didn't stick
        return {
            'success': False,
            'error': 'State transition verification failed',
            'epic_id': epic_id,
            'evidence': evidence
        }

    except Exception as e:
        # Rollback safety: log error but don't crash workflow
        return {
            'success': False,
            'error': str(e),
            'epic_id': epic_id,
            'evidence': evidence
        }
