"""
Report Generator Module

Generates comprehensive attire verification reports with student details,
findings, and recommendations.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import json

from .config import AppConfig
from .db import get_student


def generate_text_report(verification_result: Dict[str, Any], cfg: AppConfig | None = None) -> str:
	"""Generate a formatted text report from verification result"""
	cfg = cfg or AppConfig()
	
	result = verification_result
	student_id = result.get("student_id", "UNKNOWN")
	student_name = result.get("student_name", "Unknown")
	gender = result.get("gender", "unknown").title()
	student_class = result.get("class", "Unknown")
	department = result.get("department", "Unknown")
	zone = result.get("zone", "Unknown")
	timestamp = result.get("timestamp", datetime.now().isoformat())
	policy = result.get("day_policy", "uniform").upper()
	status = result.get("status", "UNKNOWN")
	overall_score = result.get("overall_score", 0.0)
	
	violations = result.get("violations", {})
	total_violations = violations.get("total_violations", 0)
	critical = violations.get("critical", 0)
	high = violations.get("high", 0)
	medium = violations.get("medium", 0)
	violation_details = violations.get("details", [])
	
	# Parse timestamp
	try:
		dt = datetime.fromisoformat(timestamp)
		date_str = dt.strftime("%d/%m/%Y")
		time_str = dt.strftime("%H:%M:%S")
	except:
		date_str = "N/A"
		time_str = "N/A"
	
	# Build report
	report = f"""
╔═══════════════════════════════════════════════════════════════════════╗
║           STUDENT ATTIRE VERIFICATION REPORT                         ║
║                      {cfg.college_name.center(50)}║
╚═══════════════════════════════════════════════════════════════════════╝

REPORT DETAILS
───────────────────────────────────────────────────────────────────────
Date:                           {date_str}
Time:                           {time_str}
Verification Zone:              {zone}
College:                        {cfg.college_name}
College Address:                {cfg.college_address}

STUDENT INFORMATION
───────────────────────────────────────────────────────────────────────
Student ID:                     {student_id}
Name:                           {student_name}
Gender:                         {gender}
Class:                          {student_class}
Department:                     {department}

VERIFICATION RESULTS
───────────────────────────────────────────────────────────────────────
Overall Status:                 {status}
Compliance Score:               {overall_score:.1%}
Policy Type:                    {policy}
Current Day:                    {datetime.now().strftime('%A')}

VIOLATIONS SUMMARY
───────────────────────────────────────────────────────────────────────
Total Violations:               {total_violations}
  Critical:                     {critical}
  High:                         {high}
  Medium:                       {medium}
"""
	
	# Add detailed violations
	if total_violations > 0:
		report += "\nDETAILED VIOLATIONS\n───────────────────────────────────────────────────────────────────────\n"
		
		for i, violation in enumerate(violation_details, 1):
			severity = violation.get("severity", "medium").upper()
			item = violation.get("item", "Unknown")
			detected = violation.get("detected", "N/A")
			score = violation.get("score", 0.0)
			
			severity_symbol = {
				"CRITICAL": "🔴",
				"HIGH": "🟠",
				"MEDIUM": "🟡",
				"LOW": "🟢"
			}.get(severity, "⚪")
			
			report += f"\n{i}. [{severity_symbol}] {item}\n"
			report += f"   Severity:       {severity}\n"
			report += f"   Detected:       {detected}\n"
			report += f"   Compliance:     {score:.1%}\n"
	else:
		report += "\n✅ NO VIOLATIONS DETECTED\n───────────────────────────────────────────────────────────────────────\n"
	
	# Add recommendations
	recommendations = result.get("recommendations", [])
	if recommendations:
		report += "\nRECOMMENDATIONS\n───────────────────────────────────────────────────────────────────────\n"
		for rec in recommendations:
			report += f"• {rec}\n"
	
	# Add security checks
	security = result.get("security_checks", {})
	if security:
		report += "\nSECURITY CHECKS\n───────────────────────────────────────────────────────────────────────\n"
		authorized = security.get("authorized", {})
		time_check = security.get("time_check", {})
		emergency = security.get("emergency_check", {})
		
		auth_status = "✅ PASS" if authorized.get("status") == "OK" else "❌ FAIL"
		time_status = "✅ PASS" if time_check.get("status") == "OK" else "⚠️ CHECK"
		emerg_status = "✅ PASS" if emergency.get("status") == "OK" else "🚨 ALERT"
		
		report += f"Authorization:          {auth_status}\n"
		report += f"Entry Time:             {time_status}\n"
		report += f"Emergency Alert:        {emerg_status}\n"
	
	report += "\n" + "="*73 + "\n"
	report += "Generated by Student Attire Verification System\n"
	report += f"Report ID: {result.get('event_id', 'N/A')}\n"
	report += "="*73 + "\n"
	
	return report


def generate_html_report(verification_result: Dict[str, Any], cfg: AppConfig | None = None) -> str:
	"""Generate an HTML report from verification result"""
	cfg = cfg or AppConfig()
	
	result = verification_result
	student_id = result.get("student_id", "UNKNOWN")
	student_name = result.get("student_name", "Unknown")
	gender = result.get("gender", "unknown").title()
	student_class = result.get("class", "Unknown")
	department = result.get("department", "Unknown")
	zone = result.get("zone", "Unknown")
	timestamp = result.get("timestamp", datetime.now().isoformat())
	policy = result.get("day_policy", "uniform").upper()
	status = result.get("status", "UNKNOWN")
	overall_score = result.get("overall_score", 0.0)
	
	violations = result.get("violations", {})
	total_violations = violations.get("total_violations", 0)
	critical = violations.get("critical", 0)
	high = violations.get("high", 0)
	medium = violations.get("medium", 0)
	violation_details = violations.get("details", [])
	
	# Parse timestamp
	try:
		dt = datetime.fromisoformat(timestamp)
		date_str = dt.strftime("%d/%m/%Y")
		time_str = dt.strftime("%H:%M:%S")
	except:
		date_str = "N/A"
		time_str = "N/A"
	
	# Status badge color
	status_color = {
		"PASS": "#28a745",
		"WARNING": "#ffc107",
		"FAIL": "#dc3545"
	}.get(status, "#6c757d")
	
	html = f"""
<!DOCTYPE html>
<html>
<head>
	<meta charset="UTF-8">
	<title>Attire Verification Report - {student_id}</title>
	<style>
		body {{
			font-family: Arial, sans-serif;
			max-width: 800px;
			margin: 0 auto;
			padding: 20px;
			background-color: #f5f5f5;
		}}
		.container {{
			background-color: white;
			padding: 30px;
			border-radius: 8px;
			box-shadow: 0 2px 10px rgba(0,0,0,0.1);
		}}
		.header {{
			text-align: center;
			border-bottom: 3px solid #007bff;
			padding-bottom: 20px;
			margin-bottom: 20px;
		}}
		.header h1 {{
			margin: 0;
			color: #333;
		}}
		.header p {{
			margin: 5px 0;
			color: #666;
		}}
		.section {{
			margin-bottom: 25px;
		}}
		.section-title {{
			font-size: 18px;
			font-weight: bold;
			color: #007bff;
			border-bottom: 2px solid #e0e0e0;
			padding-bottom: 8px;
			margin-bottom: 12px;
		}}
		.info-row {{
			display: flex;
			padding: 8px 0;
			border-bottom: 1px solid #f0f0f0;
		}}
		.info-label {{
			font-weight: bold;
			width: 35%;
			color: #333;
		}}
		.info-value {{
			width: 65%;
			color: #666;
		}}
		.status-badge {{
			display: inline-block;
			padding: 10px 20px;
			background-color: {status_color};
			color: white;
			border-radius: 5px;
			font-weight: bold;
			font-size: 16px;
		}}
		.metrics {{
			display: flex;
			gap: 15px;
			margin-top: 15px;
		}}
		.metric {{
			flex: 1;
			background-color: #f8f9fa;
			padding: 15px;
			border-radius: 5px;
			text-align: center;
			border-left: 4px solid #007bff;
		}}
		.metric-value {{
			font-size: 24px;
			font-weight: bold;
			color: #007bff;
		}}
		.metric-label {{
			font-size: 12px;
			color: #666;
			margin-top: 5px;
		}}
		.violation {{
			background-color: #fff3cd;
			border-left: 4px solid #ffc107;
			padding: 12px;
			margin-bottom: 10px;
			border-radius: 4px;
		}}
		.violation.critical {{
			background-color: #f8d7da;
			border-left-color: #dc3545;
		}}
		.violation.high {{
			background-color: #ffe5cc;
			border-left-color: #fd7e14;
		}}
		.violation-item {{
			font-weight: bold;
			margin-bottom: 5px;
		}}
		.violation-details {{
			font-size: 12px;
			color: #555;
		}}
		.success {{
			background-color: #d4edda;
			border-left: 4px solid #28a745;
			padding: 12px;
			border-radius: 4px;
			color: #155724;
		}}
		.recommendation {{
			padding: 8px 12px;
			margin-bottom: 8px;
			background-color: #e7f3ff;
			border-left: 4px solid #007bff;
			border-radius: 4px;
		}}
		.footer {{
			text-align: center;
			margin-top: 30px;
			padding-top: 20px;
			border-top: 2px solid #e0e0e0;
			color: #666;
			font-size: 12px;
		}}
		@media print {{
			body {{
				background-color: white;
			}}
			.container {{
				box-shadow: none;
				padding: 0;
			}}
		}}
	</style>
</head>
<body>
	<div class="container">
		<div class="header">
			<h1>📋 ATTIRE VERIFICATION REPORT</h1>
			<p><strong>{cfg.college_name}</strong></p>
			<p>{cfg.college_address}</p>
			<p>Report Generated: {date_str} at {time_str}</p>
		</div>
		
		<div class="section">
			<div class="section-title">👤 STUDENT INFORMATION</div>
			<div class="info-row">
				<div class="info-label">Student ID:</div>
				<div class="info-value">{student_id}</div>
			</div>
			<div class="info-row">
				<div class="info-label">Name:</div>
				<div class="info-value">{student_name}</div>
			</div>
			<div class="info-row">
				<div class="info-label">Gender:</div>
				<div class="info-value">{gender}</div>
			</div>
			<div class="info-row">
				<div class="info-label">Class:</div>
				<div class="info-value">{student_class}</div>
			</div>
			<div class="info-row">
				<div class="info-label">Department:</div>
				<div class="info-value">{department}</div>
			</div>
			<div class="info-row">
				<div class="info-label">Verification Zone:</div>
				<div class="info-value">{zone}</div>
			</div>
		</div>
		
		<div class="section">
			<div class="section-title">📊 VERIFICATION RESULTS</div>
			<div style="margin-bottom: 15px;">
				<span class="status-badge">{status}</span>
			</div>
			<div class="metrics">
				<div class="metric">
					<div class="metric-value">{overall_score:.1%}</div>
					<div class="metric-label">Compliance</div>
				</div>
				<div class="metric">
					<div class="metric-value">{policy}</div>
					<div class="metric-label">Policy</div>
				</div>
				<div class="metric">
					<div class="metric-value">{total_violations}</div>
					<div class="metric-label">Violations</div>
				</div>
			</div>
		</div>
"""
	
	# Add violations section
	if total_violations > 0:
		html += f"""
		<div class="section">
			<div class="section-title">🚨 VIOLATIONS DETECTED ({total_violations})</div>
"""
		for violation in violation_details:
			severity = violation.get("severity", "medium").lower()
			item = violation.get("item", "Unknown")
			detected = violation.get("detected", "N/A")
			score = violation.get("score", 0.0)
			
			html += f"""
			<div class="violation {severity}">
				<div class="violation-item">{item}</div>
				<div class="violation-details">
					Severity: <strong>{severity.upper()}</strong> | 
					Detected: {detected} | 
					Compliance: {score:.1%}
				</div>
			</div>
"""
		html += "</div>"
	else:
		html += """
		<div class="section">
			<div class="success">✅ <strong>No violations detected</strong> - Student is compliant with dress code</div>
		</div>
"""
	
	# Add recommendations
	recommendations = result.get("recommendations", [])
	if recommendations:
		html += """
		<div class="section">
			<div class="section-title">💡 RECOMMENDATIONS</div>
"""
		for rec in recommendations:
			html += f'<div class="recommendation">{rec}</div>\n'
		html += "</div>"
	
	html += f"""
		<div class="footer">
			<p><strong>Report ID:</strong> {result.get('event_id', 'N/A')}</p>
			<p>This is an automated report generated by the Student Attire Verification System</p>
		</div>
	</div>
</body>
</html>
"""
	
	return html


def save_report(verification_result: Dict[str, Any], format_type: str = "html", cfg: AppConfig | None = None) -> Optional[Path]:
	"""Save report to file"""
	cfg = cfg or AppConfig()
	
	student_id = verification_result.get("student_id", "unknown")
	event_id = verification_result.get("event_id", "unknown")
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	
	# Ensure reports directory exists
	cfg.reports_dir.mkdir(parents=True, exist_ok=True)
	
	if format_type == "html":
		report_content = generate_html_report(verification_result, cfg)
		filename = f"report_{student_id}_{timestamp}.html"
	elif format_type == "txt":
		report_content = generate_text_report(verification_result, cfg)
		filename = f"report_{student_id}_{timestamp}.txt"
	elif format_type == "json":
		report_content = json.dumps(verification_result, indent=2, default=str)
		filename = f"report_{student_id}_{timestamp}.json"
	else:
		return None
	
	report_path = cfg.reports_dir / filename
	
	with open(report_path, "w") as f:
		f.write(report_content)
	
	return report_path
