# zhdev Predict (mirofish fork) — Planning

> Fork von [666ghj/MiroFish](https://github.com/666ghj/MiroFish) unter AGPL-3.0.
> UI-Branding: **zhdev Predict**. Engine unverändert.
> Deployment: Coolify → `dev.mirofish.zhdev.de` (Prod-Subdomain optional später).

---

## Architektur-Entscheidungen

| Schicht | Entscheidung | Begründung |
|---|---|---|
| **Variante** | Cloud (Original-Fork, kein Offline-Stack) | Server CX33 hat 8 GB RAM; Offline-Fork (Ollama+Neo4j) bräuchte 16+ GB |
| **LLM-Provider** | apigen (`https://apigen.zhdev.de/v1`) | Free-Tier Gemini/Groq/Mistral via OpenAI-kompatiblem Endpoint |
| **Vector Memory** | Zep Cloud (Free-Tier) | MiroFish ist nativ darauf gebaut; apigen hat noch keinen Embeddings-Endpoint |
| **Container** | `ghcr.io/666ghj/mirofish:latest` (vorgebaut) | Kein eigener Build → schnellerer Deploy, weniger CI |
| **Deployment** | Coolify Docker-Compose-Resource | Standard für Custom-Compose-Apps |
| **Subdomain** | `dev.mirofish.zhdev.de` (Wildcard-DNS greift automatisch) | zhdev-Konvention |

---

## AGPL-3.0 Compliance

- [x] Repo `zhdev-de/mirofish` ist **public**
- [x] Default-Branch ist `dev` (zhdev-Konvention)
- [x] `LICENSE` (AGPL-3.0) original erhalten
- [x] `NOTICE.md` mit Original-Acknowledgment angelegt
- [x] README-Header verweist auf Source (this repo)
- [x] Frontend-Nav verlinkt auf Source-Repo (`Home.vue`, "View source on GitHub")
- [x] UI-Branding "zhdev Predict" — kein Trademark-Risiko mit "MiroFish"

---

## Tech-Stack (vom Upstream übernommen)

- Backend: Python 3.11–3.12 (FastAPI o.ä.) — Port `5001`
- Frontend: Vue 3 + Vite — Port `3000`
- Image: `ghcr.io/666ghj/mirofish:latest`
- External Services: apigen (LLM), Zep Cloud (Memory)

---

## Phasen

### Phase 1 — Fork & Setup
- [x] Fork `666ghj/MiroFish` → `zhdev-de/mirofish` (public, AGPL-3.0)
- [x] Lokal nach `/root/projects/mirofish` clonen
- [x] `dev` Branch anlegen, als Default setzen
- [x] Repo-Lowercase-Rename (`MiroFish` → `mirofish`)

### Phase 2 — Branding (minimal-invasiv, AGPL-konform)
- [x] `index.html` Title + Meta
- [x] `Home.vue` nav-brand → "ZHDEV PREDICT"
- [x] `Home.vue` nav-link → `github.com/zhdev-de/mirofish`
- [x] `locales/en.json` + `locales/zh.json` Brand-Strings
- [x] `NOTICE.md` mit AGPL-Acknowledgment
- [x] README-Fork-Header

### Phase 3 — Coolify-Deployment (BLOCKIERT)
- [ ] **Christian: neuen Coolify API-Token in /root/.env eintragen** (aktuelles ist 401)
- [ ] **Christian: Zep Cloud Account erstellen** → `ZEP_API_KEY`
- [ ] **Christian: in Coolify UI Docker-Compose-Resource anlegen** ODER Token bereitstellen, damit ich es via API mache
- [ ] Resource: GitHub Repo `zhdev-de/mirofish`, Branch `dev`, Compose-File `docker-compose.yml`
- [ ] Domain-Routing klären (siehe Hinweis unten)
- [ ] Env-Vars setzen (siehe unten)
- [ ] Deploy auslösen

**Domain-Routing-Hinweis:** Frontend (Port 3000) macht API-Calls zum Backend (Port 5001). `frontend/src/api/index.js` Default ist `http://localhost:5001` — funktioniert nur, wenn beide Prozesse im selben Container auf localhost erreichbar sind UND der Browser darauf zugreift, was im Production-Setup nicht stimmt. Zwei Optionen:

1. **Zwei Domains (einfacher in Coolify):** `dev.mirofish.zhdev.de` → Service `mirofish` Port 3000, `api.dev.mirofish.zhdev.de` → Service `mirofish` Port 5001. Env: `VITE_API_BASE_URL=https://api.dev.mirofish.zhdev.de`. Achtung: VITE_-Vars werden zur Build-Zeit eingebaut — bei vorgebautem Image evtl. nicht möglich, dann eigener Build nötig (Coolify Build-Pack: Dockerfile statt Compose).
2. **Reverse-Proxy in Compose:** nginx-Service ergänzen, `/` → Frontend, `/api` → Backend. Ein Domain-Eintrag reicht.

Option 2 ist robuster für vorgebautes Image. Vorschlag: Compose-File um nginx erweitern, dann eine Domain `dev.mirofish.zhdev.de`.

### Phase 4 — Smoke-Test
- [ ] `https://dev.mirofish.zhdev.de` erreichbar
- [ ] Mini-Simulation (≤ 10 Agents, ≤ 20 Runden) starten
- [ ] apigen-Logs auf Quota prüfen (`zhdev logs apigen`)

### Phase 5 — Quota-Schutz (optional, später)
- [ ] apigen `daily_quota` für mirofish-App auf konservativen Wert setzen
- [ ] Frontend-Hint einbauen: "Limit: X Simulationen/Tag"

### Phase 6 — Prod-Deployment (optional, nach Dev-Validierung)
- [ ] `main` Branch + zweite Coolify-App `mirofish.zhdev.de`

---

## Environment Variables

Zu setzen in Coolify (Service `mirofish`):

| Var | Wert | Quelle |
|---|---|---|
| `LLM_API_KEY` | `agk_...` (apigen App-Key) | apigen Admin API |
| `LLM_BASE_URL` | `https://apigen.zhdev.de/v1` | fix |
| `LLM_MODEL_NAME` | `smart` | apigen tier alias |
| `ZEP_API_KEY` | `z_...` | Zep Cloud Free-Tier (Christian) |

apigen App-Key wird per Admin-API erstellt:
```bash
ADMIN=$(grep "^GATEWAY_ADMIN_KEY=" /data/coolify/applications/hl5ka113925dotga2vy6b7zt/.env | cut -d= -f2)
curl -X POST https://apigen.zhdev.de/v1/admin/apps \
  -H "Authorization: Bearer $ADMIN" \
  -H "Content-Type: application/json" \
  -d '{"name":"mirofish","slug":"mirofish","tier_default":"smart","daily_quota":1000}'
unset ADMIN
```
Output enthält den `key` — direkt in Coolify-Env-Var pasten, nirgendwo zwischenspeichern.

---

## Bekannte Risiken

- **Free-Tier-Quota**: 1500 RPD Gemini Flash / 14400 RPD Groq 8B. Eine Mini-Simulation (10 Agenten × 20 Runden × ~3 Calls) frisst schon ~600 Calls. Realistisch 1–2 Demos/Tag, dann Cooldown. Daily-Quota-Cap auf apigen-Seite einbauen, damit eine durchgedrehte Simulation nicht alles auffrisst.
- **Zep Free-Tier-Limits**: Müssen wir in der Praxis beobachten.
- **Embeddings via apigen**: Aktuell nicht implementiert. Wenn Zep nicht reicht, müsste apigen um `/v1/embeddings` erweitert werden.
- **Trademark "MiroFish"**: Branding bewusst "zhdev Predict", nicht "MiroFish by zhdev".

---

## Status

**Phase 1 + 2 lokal fertig, nicht gepusht.** Warte auf Christian:
1. Coolify-API-Token erneuern → ich mache Phase 3 autonom, oder
2. Coolify-UI-Klicks selbst — ich liefere Klick-Anleitung
3. Zep Cloud Account erstellen → `ZEP_API_KEY` an mich
