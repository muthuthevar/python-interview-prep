# System Design - Interview Questions

## System Design Fundamentals

### 1. Scalability
- **Horizontal Scaling** (Scale Out)
  - Add more machines/servers
  - Load balancing
  - Stateless services

- **Vertical Scaling** (Scale Up)
  - Add more resources to existing machine
  - CPU, RAM, storage upgrades
  - Limited by hardware constraints

### 2. Load Balancing
- **Types**
  - Round Robin
  - Least Connections
  - IP Hash
  - Weighted Round Robin

- **Load Balancer Types**
  - Layer 4 (Transport Layer)
  - Layer 7 (Application Layer)

### 3. Caching
- **Cache Strategies**
  - Cache-Aside (Lazy Loading)
  - Write-Through
  - Write-Back (Write-Behind)
  - Refresh-Ahead

- **Cache Invalidation**
  - Time-based expiration
  - Event-based invalidation
  - Cache stampede prevention

- **Caching Layers**
  - Application-level cache (Redis, Memcached)
  - CDN (Content Delivery Network)
  - Browser cache

### 4. Database Design
- **SQL vs NoSQL**
  - SQL: ACID, structured data, complex queries
  - NoSQL: Flexible schema, horizontal scaling, high availability

- **Database Scaling**
  - Read Replicas
  - Sharding (Horizontal Partitioning)
  - Vertical Partitioning
  - Federation

- **CAP Theorem**
  - Consistency
  - Availability
  - Partition Tolerance
  - Can only guarantee 2 out of 3

### 5. Message Queues
- **Purpose**
  - Decouple services
  - Asynchronous processing
  - Rate limiting
  - Reliability

- **Examples**
  - RabbitMQ
  - Apache Kafka
  - Amazon SQS
  - Redis Pub/Sub

### 6. Microservices vs Monolith
- **Monolith**
  - Single deployable unit
  - Easier to develop initially
  - Harder to scale

- **Microservices**
  - Independent services
  - Easier to scale
  - More complex to manage
  - Service communication overhead

## Common System Design Questions

### 1. Design a URL Shortener (like bit.ly)
**Requirements:**
- Shorten long URLs
- Redirect short URLs to original
- High availability
- Analytics

**Design:**
- **Encoding**: Base62 encoding (0-9, a-z, A-Z)
- **Database**: Key-value store (short URL -> long URL)
- **Caching**: Redis for hot URLs
- **Load Balancer**: Distribute traffic
- **Storage**: SQL for analytics, NoSQL for mappings

**Scale Considerations:**
- 100M URLs/day = ~1,200 URLs/sec
- 6 characters = 56.8 billion unique URLs
- Sharding by hash of short URL

### 2. Design a Distributed Cache
**Requirements:**
- Get/Put operations
- High availability
- Eviction policy (LRU)

**Design:**
- **Consistent Hashing**: Distribute keys across nodes
- **Replication**: Each key stored on N nodes
- **Client-side routing**: Hash key to determine node
- **Failure handling**: Replicate to next N nodes

### 3. Design a Chat System (like WhatsApp)
**Requirements:**
- One-on-one messaging
- Group messaging
- Message delivery status
- Real-time delivery

**Design:**
- **WebSocket**: Real-time bidirectional communication
- **Message Queue**: Store messages temporarily
- **Database**: Store message history
- **Push Notifications**: For offline users
- **Load Balancer**: WebSocket connection distribution

**Components:**
- Chat Service (WebSocket server)
- Message Queue (Kafka/RabbitMQ)
- Database (Cassandra for messages)
- Notification Service

### 4. Design a Rate Limiter
**Requirements:**
- Limit requests per user/IP
- Multiple rate limit strategies
- Distributed system

**Algorithms:**
- **Token Bucket**: Tokens added at fixed rate
- **Leaky Bucket**: Fixed output rate
- **Fixed Window**: Count requests in time window
- **Sliding Window**: More accurate than fixed window

**Implementation:**
- Redis for distributed counting
- Sliding window log or counter

### 5. Design a Search Engine
**Requirements:**
- Index web pages
- Fast search
- Rank results

**Components:**
- **Crawler**: Fetch web pages
- **Indexer**: Build inverted index
- **Ranking**: PageRank, TF-IDF
- **Query Processor**: Parse and execute queries

**Storage:**
- Inverted index (word -> list of documents)
- Distributed storage (shard by document ID)

### 6. Design a Notification System
**Requirements:**
- Multiple notification types (email, SMS, push)
- High throughput
- Reliable delivery

**Design:**
- **Message Queue**: Decouple notification generation from delivery
- **Worker Pools**: Process different notification types
- **Retry Logic**: Handle failures
- **Rate Limiting**: Prevent spam

### 7. Design a File Storage System (like Dropbox)
**Requirements:**
- Upload/download files
- File versioning
- Sync across devices
- Conflict resolution

**Design:**
- **Object Storage**: S3-like for file storage
- **Metadata Database**: File info, versions, permissions
- **Sync Service**: Detect and sync changes
- **CDN**: Fast file delivery

### 8. Design a Social Media Feed (like Twitter)
**Requirements:**
- Post tweets
- Follow users
- Timeline generation
- Real-time updates

**Design Approaches:**
- **Fan-out on Write**: Pre-compute timelines
- **Fan-out on Read**: Compute on demand
- **Hybrid**: Fan-out for celebrities, on-read for others

**Components:**
- **Feed Service**: Generate timelines
- **Graph Service**: Manage follow relationships
- **Timeline Cache**: Store pre-computed timelines
- **Message Queue**: Handle fan-out

## Design Patterns for Distributed Systems

### 1. Circuit Breaker
- Prevents cascading failures
- Opens circuit after threshold failures
- Allows recovery attempts

### 2. Bulkhead
- Isolate resources
- Prevent one service from consuming all resources
- Separate thread pools, connection pools

### 3. Saga Pattern
- Manage distributed transactions
- Compensating transactions for rollback
- Event-driven coordination

### 4. Event Sourcing
- Store all changes as events
- Rebuild state by replaying events
- Audit trail built-in

### 5. CQRS (Command Query Responsibility Segregation)
- Separate read and write models
- Optimize each independently
- Event-driven updates

## Performance Optimization

### 1. Database Optimization
- **Indexing**: B-tree, hash indexes
- **Query Optimization**: Explain plans, query rewriting
- **Connection Pooling**: Reuse database connections
- **Read Replicas**: Distribute read load

### 2. API Optimization
- **Pagination**: Limit result size
- **Filtering**: Reduce data transfer
- **Compression**: Gzip responses
- **CDN**: Cache static content

### 3. Caching Strategies
- **Cache Hot Data**: Frequently accessed
- **Cache Expiration**: TTL based on data freshness needs
- **Cache Warming**: Pre-populate cache
- **Cache Coherency**: Invalidate on updates

## Monitoring and Observability

### 1. Metrics
- **Application Metrics**: Response time, error rate
- **Infrastructure Metrics**: CPU, memory, disk
- **Business Metrics**: User actions, revenue

### 2. Logging
- **Structured Logging**: JSON format
- **Log Aggregation**: Centralized logging (ELK stack)
- **Log Levels**: Debug, Info, Warning, Error

### 3. Tracing
- **Distributed Tracing**: Track requests across services
- **OpenTelemetry**: Standard for observability
- **Performance Analysis**: Identify bottlenecks

## Security Considerations

### 1. Authentication & Authorization
- **OAuth 2.0**: Third-party authentication
- **JWT**: Stateless authentication
- **RBAC**: Role-based access control

### 2. Data Protection
- **Encryption**: At rest and in transit
- **Hashing**: Passwords (bcrypt, argon2)
- **Secrets Management**: Secure storage

### 3. API Security
- **Rate Limiting**: Prevent abuse
- **Input Validation**: Sanitize inputs
- **HTTPS**: Encrypted communication

## Interview Tips

### 1. Clarify Requirements
- Ask about scale (users, requests per second)
- Functional requirements
- Non-functional requirements (latency, availability)

### 2. Start High-Level
- Draw system architecture
- Identify major components
- Show data flow

### 3. Deep Dive
- Discuss specific technologies
- Explain trade-offs
- Address bottlenecks

### 4. Scale the Design
- Handle 10x, 100x traffic
- Identify bottlenecks
- Propose solutions

### 5. Discuss Trade-offs
- Consistency vs Availability
- Latency vs Throughput
- Cost vs Performance

## Common Python-Specific System Design

### 1. Python Web Application Architecture
- **WSGI/ASGI**: Web server interface
- **Gunicorn/uWSGI**: Application servers
- **Nginx**: Reverse proxy, load balancer
- **Celery**: Distributed task queue

### 2. Python Microservices
- **FastAPI/Flask**: API frameworks
- **gRPC**: Inter-service communication
- **Kafka**: Event streaming
- **Docker**: Containerization

### 3. Python Data Processing
- **Pandas**: Data manipulation
- **Dask**: Parallel computing
- **Apache Spark**: Big data processing
- **Airflow**: Workflow orchestration
