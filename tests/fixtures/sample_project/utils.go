// Package main — utility helpers for the fixture project.
package main

import "fmt"

// FormatResult formats a query result map as a human-readable string.
func FormatResult(result map[string]interface{}) string {
	return fmt.Sprintf("%v", result)
}

// ValidateSQL returns true if the SQL string is non-empty after trimming.
func ValidateSQL(sql string) bool {
	return len(sql) > 0
}

// CountFields returns the number of fields in a parsed query result.
func CountFields(result map[string]interface{}) int {
	fields, ok := result["fields"].([]string)
	if !ok {
		return 0
	}
	return len(fields)
}
