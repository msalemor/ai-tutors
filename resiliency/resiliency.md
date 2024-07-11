# Azure Resiliency

## 1.0 - Resiliency

### 1.1 - The Azure Well-Architected Framework

The Azure Well-Architected Framework is a set of guiding tenets from Microsoft that aims to help cloud architects build secure, high-quality, and efficient infrastructure and applications on the Azure platform. It provides a consistent approach to evaluating architectures and systems against established best practices and specific workload requirements.

The framework is structured around five pillars, each addressing a core area of architecture excellence:

1. **Reliability**: This pillar focuses on ensuring that a system is resilient, available, and recoverable. The goal is to design workloads that operate consistently and predictably, with strategies in place for handling failures and minimizing disruptions.

2. **Security**: Security is paramount in the framework, emphasizing the protection of data, applications, and infrastructure. It involves threat detection, mitigation, and the maintenance of confidentiality, integrity, and availability.

3. **Cost Optimization**: This involves understanding and controlling where and how money is being spent, ensuring that resources are used efficiently, and waste is minimized. It's about getting the best value out of the investment in Azure services.

4. **Operational Excellence**: Operational excellence is about maintaining and monitoring systems to ensure they're performing as expected. This includes following best practices in DevOps, such as continuous integration and deployment, and automating operations where possible.

5. **Performance Efficiency**: The final pillar deals with the ability of a system to adapt to changes in load. It involves designing systems that can scale out effectively and utilize resources in the most efficient way possible.

These pillars guide architects in designing and building systems that not only meet functional requirements but also are reliable, secure, cost-effective, operationally sound, and performant. For more detailed information, you can refer to the official Microsoft documentation on the Azure Well-Architected Framework.

### SLA, SLO, RTO, and RPO

Understanding Service Level Agreements (SLAs), Service Level Objectives (SLOs), Recovery Point Objectives (RPOs), and Recovery Time Objectives (RTOs) is crucial in the context of resiliency and high availability for several reasons:

1. **Defining Expectations**: SLAs are contractual agreements that define the expected performance and availability levels of a service. They set the groundwork for what customers can expect and what providers aim to deliver.

2. **Measuring Performance**: SLOs are specific, measurable targets within SLAs that help in quantifying the performance and reliability of a service. They allow organizations to track whether they are meeting the expectations set forth in the SLA.

3. **Planning for Recovery**: RPOs and RTOs are essential components of disaster recovery planning. The RPO defines the maximum acceptable duration of data loss during an incident, while the RTO sets the maximum acceptable time for restoring service after an incident. These metrics guide the development of strategies to minimize data loss and downtime.

4. **Financial Implications**: Not meeting the agreed-upon SLAs can have financial consequences for service providers, which may include penalties or compensations. Therefore, understanding these agreements can have direct economic implications.

5. **Risk Management**: These metrics help in assessing the risks associated with service delivery and in implementing appropriate risk mitigation strategies. They are vital for maintaining trust and ensuring customer satisfaction.

6. **Resource Allocation**: By understanding these metrics, organizations can make informed decisions about where to allocate resources to improve reliability and performance, and to ensure that recovery measures are in place and effective.

7. **Continuous Improvement**: Monitoring SLAs, SLOs, RPOs, and RTOs enables organizations to identify areas for improvement and to make continuous enhancements to their services.

In summary, these metrics are not just technical benchmarks but are also tied to business outcomes and customer experiences. They play a pivotal role in designing, operating, and improving systems to be resilient and highly available.

#### Questions

What is SLA?
What is SLO?
In Azure what is availability zones and how can they help with resiliency?

### 1.2 - SLA Calculations

A compound Service Level Agreement (SLA) calculation is a method used to determine the overall level of service or uptime that can be expected from a combination of multiple services, each with its own SLA. This is particularly relevant in cloud computing, where a single application might rely on several cloud services, each with different availability guarantees.

To calculate the compound SLA, you typically multiply the availability percentages of the individual services. This is because the overall availability is only as strong as the weakest link in the chain; if one service fails, it can potentially take down the entire application. Therefore, the compound SLA reflects the probability of all services being available simultaneously.

For example, let's consider an application that uses two cloud services: Service A with an SLA of 99.99% availability and Service B with an SLA of 99.95% availability. To calculate the compound SLA, you would multiply the availability of both services:

- Compound SLA = Availability of Service A * Availability of Service B
- Compound SLA = 99.99% * 99.95% = 99.94%

This result shows that, although each service has a high availability when combined, the overall system has a slightly lower availability due to the compounded risk of failure.

It's important to note that this calculation assumes that the failures of the individual services are independent events. In real-world scenarios, services may have dependencies or common failure points, which can affect the actual compound SLA. Additionally, architects and engineers can design systems with redundancy and failover mechanisms to improve the compound SLA, such as using a traffic manager that routes around failed components or having backup services in different regions.

Understanding compound SLAs is crucial for businesses to evaluate the reliability of their cloud-based applications and to make informed decisions about their architecture and service providers. It also helps in setting realistic expectations for application performance and in planning for potential downtime.

#### Questions:

- Provide a sample SLA calculation for services in one region.
- Provide a sample SLA calculation for services in two load balanced regions.

### 1.3 - Architecture Patterns

The choice between these architectures depends on the specific requirements of the application, including the desired level of resiliency, the geographic distribution of the user base, and the budget for infrastructure costs. For more detailed guidance, Microsoft Azure provides extensive documentation and best practices for designing highly available architectures.

**Single Region:**
- **Resiliency & Availability:** Offers basic resiliency within a single geographic location.
- **Cost:** Generally lower due to the absence of multi-region replication.
- **Implementation & Administration:** Simpler to implement and manage.
- **Use Cases:** Suitable for applications with a regional user base and less stringent availability requirements.

**Single Region with Availability Zones:**
- **Resiliency & Availability:** Higher than single region as it distributes resources across physically separate zones within the same region.
- **Cost:** Higher than single region due to additional infrastructure but less than multi-region setups.
- **Implementation & Administration:** More complex than a single region but simpler than multi-region architectures.
- **Use Cases:** Ideal for critical applications that require high availability and can tolerate some latency within a single region.

**Passive-Active:**
- **Resiliency & Availability:** The passive region provides a failover option, enhancing resiliency.
- **Cost:** Higher due to standby resources maintained in the passive region.
- **Implementation & Administration:** Requires synchronization and failover mechanisms between regions.
- **Use Cases:** Good for applications that need regional failover capabilities without the need for simultaneous multi-region presence.

**Active-Active:**
- **Resiliency & Availability:** Maximizes availability by running workloads concurrently in multiple regions.
- **Cost:** Highest due to running full production workloads in multiple regions.
- **Implementation & Administration:** Most complex due to the need for load balancing and data replication across regions.
- **Use Cases:** Best for mission-critical applications with a global user base and the need for the highest availability and disaster recovery capabilities.

#### Questions

In Azure, which services support replication?
In Azure, which services support availability zones?

### 1.4 - Zonal Service Types

Azure services that support availability zones are designed to ensure high availability and resilience for your applications. Availability zones are unique physical locations within an Azure region, each with independent power, cooling, and networking. By architecting your solutions across multiple availability zones, you can protect your applications and data from datacenter failures.

Azure offers three types of services that support availability zones:

1. **Zonal Services**: These services enable you to pin resources to a specific zone, which can be beneficial for meeting strict latency or performance requirements. For instance, you can deploy virtual machines, managed disks, or IP addresses to a particular zone.

2. **Zone-Redundant Services**: These services automatically replicate your resources across zones without the need for manual intervention. This replication ensures that a failure in one zone does not impact the continuous availability of the service.

3. **Always-Available Services**: These services are available across all Azure geographies and are resilient to both zone-wide and region-wide outages.

It's important to note that while many Azure services support availability zones, some may have limitations based on the region, tier, or SKU. For a detailed list of services and their availability zone support, you can refer to the official Microsoft documentation on Azure services that support availability zones.

Additionally, Azure is continuously expanding its global footprint, adding new regions and availability zones to provide customers with more options for deploying their services. To stay updated on the latest regions that support availability zones, you can check the Azure geographies page.

For specific guidance on service reliability using availability zones and recommended disaster recovery strategies, the Azure reliability overview provides comprehensive information. This resource can help you design a robust reliability strategy that leverages the full capabilities of Azure's availability zones.

## 2.0 - Application Resiliency

### 2.1 - Resiliency Patterns

Resiliency patterns in software development are design strategies used to ensure that a system remains operational and responsive, even in the face of errors, failures, or unexpected conditions. Here are some common resiliency patterns:

1. **Retry Pattern**: This involves repeating a failed operation with the hope that the error was transient and the repeat attempt will be successful.

2. **Circuit Breaker Pattern**: To prevent a system from repeatedly trying to execute an operation that's likely to fail, this pattern temporarily halts operations for a particular service.

3. **Bulkhead Pattern**: This pattern isolates elements into compartments, similar to a ship's bulkhead, to prevent failures from cascading throughout the system.

4. **Timeout Pattern**: Operations are given a time limit to prevent system resources from being held indefinitely, which could lead to system failure.

5. **Fallback Pattern**: Provides alternative solutions or responses when a primary method fails, ensuring system functionality remains intact.

6. **Cache Pattern**: Storing a copy of frequently accessed data or computations to avoid repeated operations, which can improve resilience by reducing dependencies on external systems.

7. **Throttling Pattern**: Regulating the number of requests to a system to prevent overload and maintain system stability.

#### Questions

How can application architecture help with reliability?

#### References

- [Azure - Cloud design patterns that support reliability](https://learn.microsoft.com/en-us/azure/well-architected/reliability/design-patterns)

### 2.2 - Monitoring

Monitoring plays a pivotal role in enhancing the reliability of cloud architecture. It provides a continuous assessment of system performance, enabling the early detection and resolution of issues that could lead to system downtime or degradation of service quality. Here's a detailed look at how monitoring contributes to reliability in cloud architecture:

1. **Proactive Issue Identification**: Monitoring tools can detect anomalies and performance issues in real-time. This allows for proactive measures to be taken before these issues escalate into more significant problems that could affect users.

2. **Performance Benchmarks**: By establishing performance benchmarks, monitoring helps in maintaining the expected service levels. It ensures that the cloud services are performing within the set thresholds and adheres to Service Level Agreements (SLAs).

3. **Resource Optimization**: Monitoring provides insights into resource utilization, which is crucial for scaling resources up or down based on demand. This ensures that the system remains stable and responsive under varying load conditions.

4. **Automated Alerts and Responses**: Automated alerting systems can notify the relevant personnel or trigger automated workflows to respond to and resolve incidents quickly, often before users are impacted.

5. **Disaster Recovery**: In the event of a failure, monitoring tools can facilitate a quicker recovery by pinpointing the issue's source, thereby reducing the Recovery Time Objective (RTO) and minimizing service disruption.

6. **Security Assurance**: Continuous monitoring of cloud infrastructure can also detect security threats or breaches, allowing for immediate action to protect data and maintain trust in cloud services.

7. **Compliance Tracking**: For industries that require strict compliance with regulatory standards, monitoring tools can ensure that the cloud architecture is adhering to the necessary guidelines and protocols.

8. **Data-Driven Decision Making**: The data collected through monitoring can be analyzed to make informed decisions about architecture improvements, capacity planning, and investment in infrastructure.

### 2.3 - Infrastructure-as-Code (IaC)

Infrastructure as Code (IaC) is a key practice in cloud architecture that significantly enhances both resiliency and reliability. Here's how:

**Resiliency through IaC:**
1. **Automated Recovery:** IaC allows for the automation of infrastructure provisioning and management, which includes recovery processes. This automation ensures that if a component fails, the system can automatically redeploy the necessary resources without manual intervention, thus maintaining service continuity.
2. **Consistent Environments:** By treating infrastructure as code, environments are provisioned in a consistent manner. This consistency reduces the risk of configuration drift, which can lead to unexpected behavior or system vulnerabilities.
3. **Disaster Recovery:** IaC enables quick restoration of services by allowing infrastructure to be versioned and stored as code in repositories. In the event of a disaster, infrastructure can be quickly re-provisioned using the stored code, minimizing downtime.

**Reliability through IaC:**
1. **Elimination of Human Error:** Manual processes are prone to errors. IaC minimizes human intervention in the setup and maintenance of infrastructure, thereby reducing the potential for mistakes that can cause outages or performance issues.
2. **Reproducibility:** With IaC, the entire infrastructure setup is defined in code, which means it can be replicated exactly. This reproducibility ensures that every deployment is identical, reducing the likelihood of issues arising from differences in environments.
3. **Version Control:** Infrastructure code can be version-controlled, allowing for tracking changes, auditing, and rolling back to previous stable versions if necessary. This control enhances the stability of the environment.
