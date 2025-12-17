# Fault Code Reference

This document describes the fault labels used in the **ML Classifier for Fault Diagnosis in Rotary Machines** project.

Each vibration sample is assigned a numerical label corresponding to a specific fault condition in the rotary system.

---

## Fault Code Mapping

### Bearing (1) – Ball Fault Combinations
- **1.0** – Bearing (1) Ball & Bearing (2) Combination  
- **2.0** – Bearing (1) Ball & Bearing (2) Inner  
- **3.0** – Bearing (1) Ball & Bearing (2) Outer  
- **4.0** – Bearing (1) Ball & Shaft Fault (Centrally bent)  
- **5.0** – Bearing (1) Ball & Shaft Fault (Coupling end bent)  

---

### Bearing (1) – Combination Faults
- **6.0** – Bearing (1) Combination & Bearing (2) Ball  
- **7.0** – Bearing (1) Combination & Bearing (2) Inner  
- **8.0** – Bearing (1) Combination & Bearing (2) Outer  
- **9.0** – Bearing (1) Combination & Shaft Fault (Centrally bent)  
- **10.0** – Bearing (1) Combination & Shaft Fault (Coupling end bent)  

---

### Bearing (1) – Single Faults
- **11.0** – Bearing (1) Fault (Ball)  
- **12.0** – Bearing (1) Fault (Combination)  
- **13.0** – Bearing (1) Fault (Inner race)  
- **14.0** – Bearing (1) Fault (Outer race)  

---

### Bearing (1) – Inner Race Combinations
- **15.0** – Bearing (1) Inner & Bearing (2) Ball  
- **16.0** – Bearing (1) Inner & Bearing (2) Combination  
- **17.0** – Bearing (1) Inner & Bearing (2) Outer  
- **18.0** – Bearing (1) Inner & Shaft Fault (Coupling end bent)  
- **19.0** – Bearing (1) Inner & Shaft Fault (Centrally bent)  

---

### Bearing (1) – Outer Race Combinations
- **20.0** – Bearing (1) Outer & Bearing (2) Ball  
- **21.0** – Bearing (1) Outer & Bearing (2) Combination  
- **22.0** – Bearing (1) Outer & Bearing (2) Inner  
- **23.0** – Bearing (1) Outer & Shaft Fault (Centrally bent)  
- **24.0** – Bearing (1) Outer & Shaft Fault (Coupling end bent)  

---

### Bearing (2) – Ball & Combination Faults
- **25.0** – Bearing (2) Ball & Shaft Fault (Centrally bent)  
- **26.0** – Bearing (2) Ball & Shaft Fault (Coupling end bent)  
- **27.0** – Bearing (2) Combination & Shaft Fault (Centrally bent)  
- **28.0** – Bearing (2) Combination & Shaft Fault (Coupling end bent)  

---

### Bearing (2) – Single Faults
- **29.0** – Bearing (2) Fault (Ball)  
- **30.0** – Bearing (2) Fault (Combination)  
- **31.0** – Bearing (2) Fault (Inner race)  
- **32.0** – Bearing (2) Fault (Outer race)  

---

### Bearing (2) – Inner Race Combinations
- **33.0** – Bearing (2) Inner & Shaft Fault (Centrally bent)  
- **34.0** – Bearing (2) Inner & Shaft Fault (Coupling end bent)  

---

### Bearing (2) – Outer Race Combinations
- **35.0** – Bearing (2) Outer & Shaft Fault (Centrally bent)  
- **36.0** – Bearing (2) Outer & Shaft Fault (Coupling end bent)  

---

### Shaft Faults and No-Fault Condition
- **37.0** – No Fault  
- **38.0** – Shaft Fault (Centrally bent)  
- **39.0** – Shaft Fault (Coupling end bent)  

---

## Notes
- Fault codes are used as classification labels during model training and evaluation.
- The **No Fault (37.0)** label is used for binary fault detection tasks.
- All other labels represent specific mechanical fault combinations.
