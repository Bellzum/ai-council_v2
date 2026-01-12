#!/usr/bin/env python3
"""Update High priority bug tickets with detailed solutions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / ".claude/skills"))

from work_tracking import get_adapter

adapter = get_adapter()

# Bug #1183: No Rate Limiting - API Abuse Risk
description_1183 = """**Problem Summary**
Multiple workflow instances can modify the same state file simultaneously, causing data corruption. The `save()` method in `WorkflowState` class writes to state files without acquiring file locks.

**Impact**
- API abuse: Exceeds Azure DevOps rate limits (200 requests/minute)
- Workflow failures: Rate limit errors halt verification
- Service degradation: Impacts other users on shared Azure DevOps instance

**Location**
File: `core/state_manager.py`
Lines: 313-343 (`cross_verify_with_adapter()` method)

**Solution Implementation**

1. **Add rate limiting with configurable delay**
   ```python
   import time

   def cross_verify_with_adapter(
       self,
       adapter,
       batch_size: int = 10,
       delay_seconds: float = 1.0
   ):
       # Cross-verify state against external source with rate limiting

       work_item_ids = self._extract_work_item_ids()
       verified = []
       missing = []
       errors = []

       for i, work_item_id in enumerate(work_item_ids):
           try:
               # Verify work item exists
               work_item = adapter.get_work_item(work_item_id)
               if work_item:
                   verified.append(work_item_id)
               else:
                   missing.append(work_item_id)
           except Exception as e:
               errors.append({"work_item_id": work_item_id, "error": str(e)})

           # Rate limiting: Pause after every batch_size items
           if (i + 1) % batch_size == 0 and (i + 1) < len(work_item_ids):
               time.sleep(delay_seconds)

       return {
           "verified_count": len(verified),
           "missing_count": len(missing),
           "error_count": len(errors),
           "verified_ids": verified,
           "missing_ids": missing,
           "errors": errors
       }
   ```

2. **Batch verification optimization**
   - Check if adapter supports batch queries (e.g., Azure DevOps WIQL)
   - Use batch API calls instead of individual queries
   - Reduces API calls by 90%: 100 items to 10 batch calls

3. **Configurable rate limits**
   - Default: 10 items per batch, 1 second delay
   - Aggressive: 50 items per batch, 0.5 second delay (500 items/minute)
   - Conservative: 5 items per batch, 2 second delay (150 items/minute)

**Code Changes Required**
- File: `core/state_manager.py`
- Method: `cross_verify_with_adapter()` (lines 313-343)
- Add parameters: `batch_size`, `delay_seconds`
- Add rate limiting: `time.sleep()` after every batch
- Add optimization: Batch queries via WIQL (if adapter supports)

**Testing Plan**

1. **Unit Tests**
   - Test rate limiting: Verify delay called every N items
   - Test batch_size parameter: 10, 50, 100
   - Test delay_seconds parameter: 0.5, 1.0, 2.0
   - Test with 0 items, 1 item, 100 items, 1000 items

2. **Integration Tests**
   - Verify 100 work items with default rate limit
   - Measure API calls per minute (should be <= 60 calls/min)
   - Test with Azure DevOps adapter (actual rate limits)

3. **Performance Tests**
   - Benchmark: Time to verify 100, 500, 1000 items
   - Compare: Batch queries vs individual queries
   - Measure: API call reduction with batch optimization

4. **Edge Cases**
   - Test rate limit with very fast adapter (file-based)
   - Test rate limit with slow network (high latency)
   - Test interruption mid-batch (graceful abort)

**Acceptance Criteria**
- [ ] Rate limiting enforced (configurable batch size and delay)
- [ ] Default: 10 items/batch, 1 second delay (600 items/min max)
- [ ] Batch query optimization (if adapter supports)
- [ ] Unit tests: 100% coverage on rate limiting code
- [ ] Integration test: 100 items verified, no rate limit errors
- [ ] Performance: >=10x improvement with batch queries (100 items to 10 API calls)
"""

result = adapter.update_work_item(
    work_item_id=1183,
    fields={'System.Description': description_1183}
)
print(f"✅ Updated Bug #1183")

# Continue with other High priority bugs...
print("\n✅ Bug #1183 updated successfully")
