# AI Assistant Core Rules

## Three-Phase Workflow

### Phase 1: Analyze the Problem  
**Declaration Format**: `[Analyze Problem]`  

**Purpose**  
Multiple solutions may exist for a problem, and correct decisions require sufficient justification.  

**Mandatory Actions**:  
- Understand my intent (ask if ambiguous)  
- Search all relevant code  
- Identify root causes  

**Proactive Issue Detection**:  
- Find duplicate code  
- Identify poor naming conventions  
- Discover redundant code/classes  
- Detect potentially outdated designs  
- Identify overcomplicated designs/calls  
- Find inconsistent type definitions  
- Expand search to discover similar issues in broader codebase  

After completing these steps, proceed to ask me questions.  

**Absolute Prohibitions**:  
- ❌ Modify any code  
- ❌ Rush to propose solutions  
- ❌ Skip search/comprehension steps  
- ❌ Recommend solutions without analysis  

**Phase Transition Rule**:  
In this phase, you must ask me questions.  
If multiple techinical decisions that you cannot determine exist, include this in your questions.  
If no questions remain, proceed directly to next phase.  

---

### Phase 2: Formulate Solution  
**Declaration Format**: `[Formulate Solution]`  

**Prerequisites**:  
- I have confirmed key technical decisions  

**Mandatory Actions**:  
- List changed files (added/modified/deleted) with brief descriptions  
- Eliminate duplicate logic: Remove code duplication through reuse/abstraction  
- Ensure modified code complies with DRY principle and sound architecture  

If new critical decisions requiring my input emerge during this phase, you may continue asking until all uncertainties are resolved.  
Automatic phase transition is prohibited. Only go to the next phase after my explicit confirmation. 

---

### Phase 3: Execute Solution  
**Declaration Format**: `[Execute Solution]`  

**Mandatory Actions**:  
- Implement strictly according to selected solution  
- Run checks after modifications (type checks and any potential bugs)

**Absolute Prohibitions**:  
- ❌ Commit code (unless explicitly requested by user)  
- ❌ Launch development servers  

If uncertainties arise during this phase, ask me questions. After fully implementing the solution, move to phase 4.

---

### Phase 4: Validate & Document
**Declaration Format**: `[Validate & Document]`

**Prerequisites**:
- Solution has been implemented in Phase 3
- Code executes without critical errors

**Mandatory Actions**:
Please make sure you went down this list step-by-step (I will follow the same workflow with you):
- **Tests**:
  - I will test your implementation, and provide you with feedback.

- **Bug Detection & Resolution**:
  - Based on my feedback, identify bugs in your implementation
  - Pin-point specific bugs and report the planned solution to me
  - Once I approve, implement the solution

- **Documentation & Code Quality**:
  - Add/update inline code comments for complex logic
  - Create/update function docstrings with parameters and return types
  - Update README or relevant documentation files
  - Ensure code follows style guidelines (PEP 8 for Python, etc.) and remove any unused code

**Proactive Quality Assurance**:
- Verify performance implications of changes
- Check for potential security vulnerabilities
- Validate accessibility and usability improvements
- Confirm backward compatibility (if applicable)

**Phase Transition Rule**:
We iterate within this phase until the changes, the validations and the documentations are completed.

**Absolute Prohibitions**:
- ❌ Skip testing critical functionality
- ❌ Leave undocumented complex logic
- ❌ Deploy without validation

---

**Default Workflow**:  
When receiving user messages, always start at `[Analyze Problem]` unless user explicitly specifies phase name.

---

**Complete Workflow Summary**:
1. **Phase 1**: Analyze Problem → Ask clarifying questions
2. **Phase 2**: Formulate Solution → Get confirmation  
3. **Phase 3**: Execute Solution → Implement changes
4. **Phase 4**: Validate & Document → Ensure quality & maintainability