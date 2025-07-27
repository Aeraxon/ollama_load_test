# Ollama Load Test Tool

A realistic load testing tool for Ollama instances that simulates real user behavior and helps determine the capacity limits of your LLM chat environment.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Parameters](#parameters)
- [Examples](#examples)
- [Prompts File](#prompts-file)
- [Test Duration and Timing Behavior](#test-duration-and-timing-behavior)
- [Interpreting Results](#interpreting-results)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)

## Overview

This tool performs automated load tests by gradually increasing the number of simulated users (e.g., 5, 10, 15, 20 users). Each test runs for a defined duration (default 5 minutes) and collects detailed metrics. At the end, an evaluation table is automatically created to help identify optimal capacity.

**Key Features:**
- **Gradual User Increase**: Automatic tests from 5 to the desired maximum number
- **Multi-Model Support**: Compare multiple Ollama models in one run
- **Realistic Pause Times**: Configurable thinking pauses between prompts (3-30 seconds)
- **Automatic Evaluation**: Results table with all important metrics
- **GPU Documentation**: Storage of used hardware for later reference
- **System Monitoring**: CPU and memory monitoring during tests
- **CSV Export**: Results exportable for further analysis

## Features

### Automated Test Execution
- **Gradual Increase**: User count is automatically increased in configurable steps
- **Test Duration Behavior**: New requests are sent for the specified duration, then waits for completion of all running requests
- **Continuous Tests**: Users send requests throughout the entire active test duration
- **Automatic Termination**: Tests are terminated early if error rate exceeds 30%
- **Automatic Evaluation**: Results table is automatically created at the end

### Realistic Simulation
- **Variable Pause Times**: Realistic thinking pauses between prompts (default: 3-30 seconds)
- **Random Prompt Selection**: Prompts are used in random order
- **Customer-Specific Prompts**: Use real prompts from your work environment

### Comprehensive Metrics
- **Response Times**: Average, maximum, minimum of complete response time
- **Time-to-First-Token (TTFT)**: Average time until first token (UX-critical)
- **Error Rate**: Percentage of failed requests
- **System Monitoring**: CPU and memory usage during tests
- **Request Statistics**: Successful vs. failed requests
- **Hardware Documentation**: GPU information for later reference
- **Automatic Assessment**: Intelligent recommendations based on TTFT and stability

### Multi-Model Testing
- **Model Comparison**: Test multiple Ollama models sequentially
- **Consistent Conditions**: All models under identical test conditions
- **Clear Results**: Direct comparison of model performance

## Installation

### Prerequisites
- Python 3.7 or higher
- `requests` library
- Running Ollama instance

### Installing Dependencies
```bash
pip install requests
```

### Download Script
Save the script as `ollama_load_test.py` and make it executable:
```bash
chmod +x ollama_load_test.py
```

## Usage

### Basic Syntax
```bash
python ollama_load_test.py --prompts PROMPTS_FILE --users MAX_USERS --model MODEL(S) [OPTIONS]
```

### Minimal Example
```bash
# Tests gradually 5, 10, 15, 20, 25 users (5 minutes per step each)
python ollama_load_test.py --prompts my_prompts.txt --users 25 --model llama2

# Compares multiple models
python ollama_load_test.py --prompts my_prompts.txt --users 25 --model "llama2,mistral,codellama"
```

## Parameters

### Required Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--prompts` | Path to prompts file | `--prompts customer_prompts.txt` |
| `--users` | Maximum number of users (reached gradually) | `--users 50` |
| `--model` | Ollama model(s), comma-separated for multiple models | `--model "llama2,mistral"` |

### Optional Parameters

| Parameter | Default | Description | Example |
|-----------|---------|-------------|---------|
| `--gpu` | Unknown | GPU designation for documentation | `--gpu "RTX A2000"` |
| `--pause-min` | 3.0 | Minimum pause between messages (seconds) | `--pause-min 1.0` |
| `--pause-max` | 30.0 | Maximum pause between messages (seconds) | `--pause-max 60.0` |
| `--step-size` | 5 | Step size for user increase | `--step-size 10` |
| `--test-duration` | 300 | Duration of active request phase per step (seconds) | `--test-duration 600` |
| `--host` | 127.0.0.1:11434 | Ollama host and port | `--host 192.168.x.x:11434` |
| `--output` | Auto | CSV filename for export | `--output results.csv` |

## Examples

### Multi-Model Comparison
```bash
# Compares 3 models gradually up to 30 users
python ollama_load_test.py \
  --prompts customer_prompts.txt \
  --users 30 \
  --model "llama2,mistral,codellama" \
  --gpu "RTX A2000"
```

### Standard Load Test with GPU Documentation
```bash
# Tests 5, 10, 15, 20, 25, 30 users (5 minutes each)
python ollama_load_test.py \
  --prompts customer_prompts.txt \
  --users 30 \
  --model llama2 \
  --gpu "RTX A2000"
```

### Quick Test with Short Pause Times
```bash
# Simulates power users with 1-5 seconds pause time
python ollama_load_test.py \
  --prompts prompts.txt \
  --users 40 \
  --model "llama2,mistral" \
  --pause-min 1 \
  --pause-max 5 \
  --step-size 10 \
  --gpu "RTX 4090"
```

### Remote Server with Longer Tests
```bash
# 10-minute tests on remote server
python ollama_load_test.py \
  --prompts support_prompts.txt \
  --users 25 \
  --model codellama \
  --host 192.168.x.x:11434 \
  --test-duration 600 \
  --gpu "V100" \
  --output remote_test_results.csv
```

### Different User Types

**Power Users (fast interaction):**
```bash
python ollama_load_test.py --prompts prompts.txt --users 20 --model "llama2,mistral" --pause-min 1 --pause-max 5 --gpu "RTX 4090"
```

**Normal Office Users (standard):**
```bash
python ollama_load_test.py --prompts prompts.txt --users 30 --model llama2 --gpu "RTX A2000"
```

**Thoughtful Users (long pauses):**
```bash
python ollama_load_test.py --prompts prompts.txt --users 15 --model llama2 --pause-min 10 --pause-max 60 --gpu "RTX A2000"
```

## Prompts File

### Format
The prompts file is a simple text file with one prompt per line:

```txt
Explain the difference between Machine Learning and Deep Learning
Write Python code for a simple to-do list
What are the pros and cons of microservices?
How can I increase my productivity when working from home?
Explain the concept of REST APIs in simple terms
```

### Best Practices for Prompts

**1. Collect realistic prompts:**
- Ask your employees for typical questions
- Analyze existing chat logs
- Use different prompt lengths

**2. Cover different categories:**
- Short factual questions
- Longer explanation requests
- Code generation
- Creative tasks
- Problem solving

**3. Example of a balanced prompts file:**
```txt
What is Python?
Explain the MVC pattern in detail with examples
def fibonacci(n): # complete this function
Write an email to a customer regarding project delays
How do I optimize the performance of a MySQL database?
What does CI/CD mean?
Create a business plan for a tech startup in the AI field
Debug this JavaScript code: function add(a,b) { return a+b }
```

## Test Duration and Timing Behavior

**Important:** The specified `--test-duration` defines only the **active request phase**, not the total test duration.

### Test Phases:

#### **Phase 1: Active Request Phase (e.g., 5 minutes)**
- New requests are continuously sent
- Users start requests until the end of the specified duration
- Example: With `--test-duration 300`, new requests are started for 5 minutes

#### **Phase 2: Waiting Phase (variable duration)**  
- **No new requests** anymore
- Waits for completion of **all running requests**
- Duration depends on model performance and current load

### Timing Example:

```
00:00 - Test starts, first requests are sent
00:00-05:00 - Active phase: Continuously new requests  
04:58 - Last new request is started
05:00 - NO new requests anymore
05:00-06:15 - Waiting phase: Running requests are completed
06:15 - Test finished (Total duration: 6:15 instead of 5:00)
```

### Why Tests May Take Longer:

**With normal performance:**
- Request duration: 2-5 seconds
- Waiting phase: A few seconds
- **Total duration ≈ specified test duration**

**With high load/slow model:**
- Request duration: 30-60 seconds or more
- Waiting phase: Up to several minutes
- **Total duration can be significantly longer**

### Automatic Termination on Overload:

- **Monitoring**: Error rate is checked every 30 seconds
- **Termination criterion**: > 30% error rate with at least 10 requests
- **Immediate stop**: Test is terminated early, all processes killed

```
[Intermediate status] Requests: 15, Error rate: 35.2%
⚠️ ABORT: Error rate (35.2%) exceeds 30%!
System is overloaded - test is being aborted.
```

## Interpreting Results

The tool automatically runs multiple tests and shows progress for each step:

```
TESTING MODEL: llama2
================================
[Step 2/12] Testing 10 users with llama2...
[User 3] ✓ 4.23s (TTFT: 1.45s) - Explain the difference between...
[User 7] ✗ Timeout
[User 1] ✓ 2.45s (TTFT: 0.89s) - What is Python?...

TESTING MODEL: mistral  
================================
[Step 8/12] Testing 10 users with mistral...
[User 2] ✓ 1.89s (TTFT: 0.67s) - Write Python code for...
```

### Automatic Results Table

At the end, you get a clear overview table:

```
LOAD TEST RESULTS
======================================================================================================
Users    Model   GPU      Avg. Time  TTFT    Max. Time  Min. Time  Error Rate  CPU %   Memory %  Requests   Recommendation
5        llama2   A2000    2.34       1.12    4.12       1.23       0.0         45.2    62.1      245        ✅ Optimal
10       llama2   A2000    3.78       1.89    7.45       1.45       2.1         78.3    68.4      467        ⚠️ Unstable
15       llama2   A2000    5.23       3.45    12.34      1.67       5.4         89.1    72.3      623        ❌ Overloaded
5        mistral  A2000    1.89       0.78    3.21       0.98       0.0         38.1    58.9      267        ✅ Optimal
10       mistral  A2000    2.45       1.23    5.67       1.12       1.4         65.4    62.3      523        ✅ Good
15       mistral  A2000    3.12       2.67    8.93       1.34       3.2         72.8    67.1      587        ⚠️ Unstable
```

### Important Metrics

**Monitor Response Times:**
- **Avg. Time**: Complete response time from request to full answer
- **TTFT (Time-to-First-Token)**: Time until first token - critical for UX
- **< 2 seconds TTFT**: Optimal for user experience
- **2-5 seconds TTFT**: Good performance
- **5-10 seconds TTFT**: Acceptable
- **> 10 seconds TTFT**: Perceived as slow

**Observe Error Rate (critical for stability):**
- **0-2%**: Production-ready ✅
- **2-5%**: Unstable, still tolerable ⚠️
- **5-10%**: Overloaded ❌
- **> 10%**: Critical problems ❌

### Identifying Capacity Limits

**Finding Optimal Capacity:**
1. **Stable Performance**: Response times remain constant, error rate under 5%
2. **Beginning Overload**: Response times increase significantly
3. **Critical Overload**: Error rate over 15%, very high CPU usage

**Model Comparison:**
In the table, you can directly compare different models:
- **mistral**: Faster TTFT (0.78s vs 1.12s), better stability
- **llama2**: Slightly slower TTFT, becomes unstable at higher load
- **Optimal Capacity**: mistral handles more users with the same hardware

**Understanding Automatic Assessment:**
- **✅ Optimal/Good**: TTFT < 5s, error rate < 2%
- **✅ Acceptable**: TTFT 5-10s, stable performance
- **⚠️ Unstable/Slow**: TTFT > 10s or error rate 2-5%
- **❌ Overloaded**: Error rate 5-10%, performance issues
- **❌ Critical**: Error rate > 10%, not production-ready

**Example Interpretation:**
- **mistral with 10 users**: ✅ Good - Optimal production range
- **llama2 with 15 users**: ❌ Overloaded - Capacity limit exceeded

### CSV Export for Further Analysis

Results are automatically saved as CSV:
```
ollama_load_test_llama2_mistral_codellama_YYYYMMDD_HHMMSS.csv
```

This file contains all metrics including TTFT and GPU information for detailed analysis in Excel, Google Sheets, or other tools.

## Troubleshooting

### Common Problems and Solutions

**Problem: "Connection error - is Ollama running?"**
```bash
# Solution: Start Ollama service
ollama serve

# Alternative: Ollama as background process
nohup ollama serve &
```

**Problem: "Timeout after 120 seconds"**
- Reduce the number of concurrent users
- Use a faster model
- Check server resources

**Problem: "HTTP error 404"**
- Check model name: `ollama list`
- Download model: `ollama pull llama2`

**Problem: Very slow response times**
```bash
# Check server resources
htop
nvidia-smi  # for GPU usage

# Check Ollama logs
journalctl -u ollama -f
```

### Network Debugging

**Test Ollama reachability:**
```bash
# Local test
curl http://127.0.0.1:11434/api/tags

# Remote test  
curl http://192.168.x.x:11434/api/tags
```

**Firewall issues:**
```bash
# Open port (Ubuntu/Debian)
sudo ufw allow 11434

# Check port status
netstat -tlnp | grep 11434
```

## Best Practices

### Test Planning

**1. Gradual Increase (automatic):**
```bash
# The tool automatically runs tests with 5, 10, 15, 20, 25 users
python ollama_load_test.py --prompts prompts.txt --users 25 --model llama2
```

**2. Multi-Model Comparison:**
```bash
# Compare different models under identical conditions
python ollama_load_test.py --prompts prompts.txt --users 25 --model "llama2,mistral,codellama" --gpu "RTX A2000"

# Find the best model for your hardware
python ollama_load_test.py --prompts prompts.txt --users 30 --model "llama2,mistral" --gpu "RTX 4090"
```

**3. Different Pause Times:**
```bash
# Fast users (stress test)
python ollama_load_test.py --prompts prompts.txt --users 40 --model llama2 --pause-min 1 --pause-max 3

# Realistic users
python ollama_load_test.py --prompts prompts.txt --users 30 --model llama2 --pause-min 3 --pause-max 30

# Slow users
python ollama_load_test.py --prompts prompts.txt --users 20 --model llama2 --pause-min 30 --pause-max 120
```

### Monitoring During Tests

**Monitor server side:**
```bash
# Resource consumption
htop
iostat -x 1

# GPU usage (if available)
nvidia-smi -l 1

# Network traffic
iftop
```

**Follow Ollama logs:**
```bash
# Systemd-based installation
journalctl -u ollama -f

# Manual installation
tail -f /var/log/ollama.log
```

### Optimization Based on Results

**For poor performance:**
1. **Hardware upgrade**: More RAM, better CPU/GPU
2. **Switch model**: Use smaller, faster model
3. **Concurrent limits**: Limit maximum number of concurrent requests
4. **Load balancing**: Use multiple Ollama instances

**Adjust configuration:**
```bash
# Set Ollama environment variables
export OLLAMA_MAX_LOADED_MODELS=2
export OLLAMA_NUM_PARALLEL=4
export OLLAMA_HOST=0.0.0.0:11434
```

### Documentation of Results

The tool automatically creates a clear results table and saves all data as CSV. Example of a typical multi-model evaluation:

| Users | Model | GPU | TTFT | Avg. Time | Error Rate | CPU % | Recommendation |
|-------|-------|-----|------|-----------|------------|-------|----------------|
| 10    | llama2 | A2000 | 1.89s | 3.8s     | 2%         | 78%   | ⚠️ Unstable |
| 10    | mistral| A2000 | 1.23s | 2.4s     | 1%         | 65%   | ✅ Good     |
| 10    | codellama| A2000 | 2.45s | 4.2s   | 3%         | 82%   | ⚠️ Unstable |
| 20    | llama2 | A2000 | 4.12s | 8.4s     | 12%        | 94%   | ❌ Critical |
| 20    | mistral| A2000 | 2.67s | 4.1s     | 4%         | 78%   | ⚠️ Unstable |
| 20    | codellama| A2000 | 6.23s | 9.8s   | 15%        | 96%   | ❌ Critical |

**Conclusion:** mistral is the most performant model - stable up to 10 users, unstable but usable up to 20 users.

---

## Conclusion

This automated load testing tool helps you quickly and systematically determine the optimal capacity of your Ollama installation. Through gradual increase of user count and automatic result evaluation, you get a clear overview of performance limits and can make informed decisions for the production environment.

**Main Benefits:**
- 🚀 **Automated**: No manual tests required
- 📊 **Clear**: Clear result tables and CSV export  
- 🎯 **Realistic**: Real prompts and realistic pause times
- ⚡ **Efficient**: All tests in one run
- 🔄 **Multi-Model**: Direct comparison of different models
- 💾 **Documented**: GPU and hardware information is saved
- 📈 **Actionable**: Directly implementable capacity recommendations

For questions or problems, you can adjust the parameters and run smaller tests to adapt the tool to your specific requirements.

## ⚠️ Disclaimer

**USE AT YOUR OWN RISK**

This software is provided "as is" without warranty of any kind. The author(s) assume **NO LIABILITY** for any damages, losses, or issues that may occur from using this software, including but not limited to: data loss, hardware damage, system failures, network issues, natural disasters, cosmic events, or any other consequences whatsoever.

By using this software, you acknowledge that you use it entirely at your own risk and take full responsibility for any outcomes.

## License

This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)**.

### 🆓 Non-Commercial Use (Free)
- ✅ Personal use, education, research, open source projects

### 💼 Commercial Use License
For commercial use, open an issue in this repository or contact via GitHub.

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
