# AI Council – Agent Descriptions  
## Pharmaceutical Drug Candidate Selection (Demo)

This document defines the five AI agents participating in the **AI Council** demo for pharmaceutical new drug candidate selection.  
Each agent represents a real-world stakeholder group with distinct responsibilities, goals, and data access levels.

---

## 1. Chief Agent  
*(CTO / Chief Scientific Officer style)*

### Role
- Acts as the final decision owner across science, business, legal, and timing.
- Resolves tradeoffs when stakeholder opinions conflict.
- Takes accountability for advancement or termination decisions.

### Goals
- Select the **Top 3 drug candidates** to advance (demo scope).
- Ensure decisions are:
  - explainable
  - auditable
  - aligned with company strategy
- Manage portfolio-level risks:
  - scientific risk
  - regulatory risk
  - intellectual property risk
  - market and commercial risk

### Data Access (High-Level Only)
- Consolidated candidate scorecards.
- Summarized evidence links and risk flags from each agent.
- Decision memo history and rationale tracking.
- No direct access to raw lab notebooks or sensitive experimental logs.

### Example External Reference Data
- Clinical trial landscape: ClinicalTrials.gov API
- Safety signal monitoring: openFDA FAERS API
- Scientific literature trends: OpenAlex API

---

## 2. R&D Agent  
*(Lab Leader + Translational R&D)*

### Role
- Integrates wet lab, dry lab, and early translational evidence.
- Acts as the scientific gatekeeper for data quality, traceability, and compliance.
- Coordinates across experimental, computational, and translational teams.

### Goals
- Reduce a broad candidate pool to the **Top 10 drug candidates** based on:
  - potency and selectivity
  - ADME and toxicity signals
  - mechanism-of-action plausibility
  - reproducibility and assay quality
- Clearly communicate:
  - what is known
  - what remains uncertain
  - what data gaps exist

### Data Access
- **Internal**
  - Wet lab: assay outputs, QC metrics, batch metadata, protocols.
  - Dry lab: predicted properties, docking results, QSAR models, multi-parameter optimization outputs.
- **External**
  - Bioactivity, target, and pathway reference data.
  - Chemical identifiers and structural information.

### Example Open Datasets
- ChEMBL: bioactivity, targets, assays, activity measurements.
- PubChem bulk downloads: chemical structures, identifiers, and bioassay links.

---

## 3. Clinical and Regulatory Agent  
*(Medical + Policy Maker Lens)*

### Role
- Represents clinical feasibility and regulatory expectations.
- Translates scientific candidates into potential human use cases.
- Frames regulatory strategy and approval pathways.

### Goals
- Define:
  - target indication fit
  - clinical endpoints
  - patient population
  - differentiation versus standard of care
- Identify regulatory risks early:
  - safety concerns
  - trial design challenges
  - comparator and endpoint risks
- Recommelsnd which **3 candidates** are ready to enter a clinical package.

### Data Access
- Clinical trial landscape and competitor programs.
- Prior approvals and label precedents.
- Post-marketing safety patterns for similar mechanisms.
- Regulatory review documents and public assessment reports.

### Example External Datasets
- ClinicalTrials.gov API: registered clinical trials.
- FDA Orange Book data files: approved drugs and therapeutic equivalence.
- Drugs@FDA data files: FDA approval records.
- EMA medicine data and EPAR-related sources.
- openFDA FAERS: post-marketing adverse event reports.

---

## 4. Business Agent  
*(Commercial Strategy + Market Access)*

### Role
- Evaluates commercial viability and strategic fit.
- Assesses whether a scientifically promising candidate can succeed in the market.

### Goals
- Estimate commercial potential for each candidate.
- Identify the strongest **“why now”** market positioning.
- Flag market access and commercialization risks:
  - pricing pressure
  - reimbursement constraints
  - generic or biosimilar competition
  - payer and policy barriers

### Data Access
- Drug spending and utilization proxies.
- Pricing benchmarks and acquisition cost references.
- Disease burden and indication mapping.
- Public competitive intelligence sources.

### Example External Datasets
- CMS Medicare Part D drug spending data.
- NADAC drug acquisition cost reference data.
- ATC/DDD index toolkit for therapeutic class mapping.

---

## 5. Legal and IP Agent  
*(Patent + Freedom to Operate)*

### Role
- Owns the intellectual property and legal risk perspective.
- Ensures innovation is protectable and defensible.

### Goals
- Assess novelty and overlap with existing patents.
- Identify freedom-to-operate risks early.
- Recommend patent filing strategy and timing aligned with R&D milestones.

### Data Access
- Patent corpora and bibliographic data.
- Assignees, claim families, citation networks.
- Legal status and filing timelines.
- Prior art aligned with targets and chemical series.

### Example External Datasets
- PatentsView PatentSearch API (US patent data).
- USPTO Bulk Data Portal (bulk patent datasets).

---

## Summary

The AI Council operates as a **multi-agent decision system** where:
- Each agent contributes specialized knowledge.
- Data access is role-dependent.
- Decisions progress from scientific filtering to clinical, commercial, and legal validation.
- The Chief Agent synthesizes all inputs to make the final advancement decision.

This structure mirrors real-world pharmaceutical decision-making while remaining suitable for a demo environment.
