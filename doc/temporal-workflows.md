Ovo je kompletno ažuriran i sređen vodič prilagođen tvojoj novoj `src/temporal/` strukturi. Zadržao sam tvoj originalni ton i format, ali sam promenio putanje, dodao `client.py` logiku i organizovao "Triggers" sekciju prema tvojoj trenutnoj implementaciji.

-----

# Temporal Workflows & Activities

This guide explains what Temporal is, how it works, and how to write workflows and activities in this project.

-----

## What is Temporal?

Temporal is a system for running background processes reliably. It handles the hard parts of distributed computing — retries, failures, long-running tasks, and state management — so you do not have to implement those yourself.

Without Temporal, if you start a long background process and your server crashes halfway through, the work is lost. With Temporal, the process automatically resumes from where it stopped.

### When to use Temporal

Use Temporal when you need to:

  - Process data in the background (not during a web request).
  - Run tasks that might fail and need to retry automatically.
  - Orchestrate a sequence of steps where each step depends on the previous one.
  - Schedule recurring jobs.

-----

## Core concepts

### Activity

An **activity** is a function that does one specific piece of work. It is the smallest unit of execution in Temporal.
Examples: Fetch a webpage, parse a document, insert a row into the database, call an external API.

### Workflow

A **workflow** is a function that **orchestrates** activities. It decides what order to run them in, what to do with their results, and how to handle errors.
A workflow should contain **no business logic itself**.

### Worker

A **worker** is a process that listens for tasks from Temporal and executes them. In this project, the worker is started in `main.py` and runs continuously inside the Docker container.

### Task queue

A task queue is a named channel through which Temporal sends work to workers. In this project, we use `sandbox-task-queue`.

-----

## Project Structure

Our Temporal logic is organized inside the `src/temporal/` directory:

  - **`src/temporal/activities/`**: Definition of atomic tasks.
  - **`src/temporal/workflows/`**: Definition of orchestration logic.
  - **`src/temporal/client.py`**: Shared connection logic.
  - **`src/temporal/worker.py`**: (Optional) Specific worker setup if separated from `main.py`.
  - **`scripts/`**: Logic for starting/triggering workflows.
-----

## How execution flows

```
Client starts workflow (via Trigger)
        │
        ▼
Temporal server receives it
        │
        ▼
Worker (listening on task queue) picks it up
        │
        ▼
Workflow function runs
        │
        ├── calls Activity A
        │         │
        │         ▼
        │   Activity A executes, returns result
        │
        ├── calls Activity B with result from A
        │         │
        │         ▼
        │   Activity B executes, returns result
        │
        ▼
Workflow completes, result is returned
```

-----

## Writing an activity

An activity is a regular Python async function decorated with `@activity.defn`.

```python
# src/temporal/activities/example_activity.py
import structlog
from temporalio import activity

log = structlog.get_logger(__name__)

@activity.defn
async def example_activity(name: str) -> str:
    """Example activity that simulates processing a task."""
    log.info("executing_activity", name=name)
    result = f"Hello: {name}. Workflow executed."
    log.info("activity_completed", result=result)
    return result
```

Rules for activities:

  - Must be decorated with `@activity.defn`.
  - Must be `async` functions.
  - Should log at the start and end so you can trace execution.
  - Can raise exceptions — Temporal will retry the activity.

-----

## Writing a workflow

A workflow is a class with a method decorated with `@workflow.run`.

```python
# src/temporal/workflows/example_workflow.py
import structlog
from datetime import timedelta
from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from src.temporal.activities.example_activity import example_activity

log = structlog.get_logger(__name__)

@workflow.defn
class ExampleWorkflow:
    @workflow.run
    async def run(self, name: str) -> str:
        log.info("workflow_started", name=name)

        # Call the activity
        result = await workflow.execute_activity(
            example_activity,
            name,
            start_to_close_timeout=timedelta(seconds=10),
        )

        log.info("workflow_completed", result=result)
        return result
```

Rules for workflows:

  - Must be a class decorated with `@workflow.defn`.
  - The main method must be decorated with `@workflow.run`.
  - **Must not contain any I/O** (no database calls, no direct HTTP requests).
  - All imports of activities must be inside `with workflow.unsafe.imports_passed_through()`.

-----

## Connection Client

We use a centralized way to connect to Temporal via `src/temporal/client.py`.

```python
# src/temporal/client.py
from temporalio.client import Client
from config.settings import settings


async def get_temporal_client() -> Client:
    """Returns a connected Temporal client."""
    return await Client.connect(settings.temporal_host)
```

-----

## Registering in the worker

Every workflow and activity must be registered in `main.py` (or your worker script).

```python
# main.py
import asyncio
from temporalio.worker import Worker
from src.temporal.client import get_temporal_client
from src.temporal.workflows.example_workflow import ExampleWorkflow
from src.temporal.activities.example_activity import example_activity

async def main():
    client = await get_temporal_client()

    worker = Worker(
        client,
        task_queue="sandbox-task-queue",
        workflows=[ExampleWorkflow],
        activities=[example_activity],
    )

    print("Worker started. Listening for tasks...")
    await worker.run()

if __name__ == "__main__":
    asyncio.run(main())
```

-----

## Starting a workflow (Triggers)

Triggers are used to start workflow executions. They can be called from FastAPI, CLI, or standalone scripts.

```python
# cripts/example_trigger.py
import uuid
from temporalio.client import Client
from src.temporal.workflows.example_workflow import ExampleWorkflow

async def run_example_trigger(client: Client, name: str = "Student") -> str:
    """Triggers the ExampleWorkflow and returns the result."""
    workflow_id = f"example-{uuid.uuid4()}"

    result = await client.execute_workflow(
        ExampleWorkflow.run,
        name,
        id=workflow_id,
        task_queue="sandbox-task-queue",
    )
    return result
```

To run a trigger manually for testing:

```bash
uv run python -m src.temporal.scripts.run_example_script
```

-----

## Monitoring in the Temporal UI

Open [http://localhost:8080](https://www.google.com/search?q=http://localhost:8080).

The UI shows:

  - All workflow runs (running, completed, failed).
  - The history of each workflow — every activity call and its result.
  - Errors and stack traces when something fails.

-----

## Common mistakes

**Forgetting to register an activity in the worker**
The workflow will hang in the UI with status "Running". Always check the `Worker` initialization in `main.py`.

**Putting I/O inside a workflow**
All I/O must go inside activities. Workflows must be deterministic for replay safety.

**Missing `with workflow.unsafe.imports_passed_through()`**
Temporal sandboxes workflow code. Imports of activities or external libraries at the module level must be inside this block.

**Using `time.sleep()` in a workflow**
Always use `await asyncio.sleep()` or `workflow.sleep()`.

-----

## Step-by-step checklist for adding a new workflow

1.  Create activity functions in `src/temporal/activities/`.
2.  Create a workflow class in `src/temporal/workflows/`.
3.  Import activities inside `with workflow.unsafe.imports_passed_through()`.
4.  Add the workflow class and activity functions to the `Worker` in `main.py`.
5.  Create a trigger in `scripts/` to start the workflow.
6.  Check [http://localhost:8080](https://www.google.com/search?q=http://localhost:8080) to see it run.