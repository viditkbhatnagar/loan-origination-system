"""
Association Rule Mining for Loan Origination
Enhanced persona calculation using frequent patterns and rules
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

class AssociationMiner:
    """Association Rule Mining for loan patterns and persona discovery"""
    
    def __init__(self, min_support=0.1, min_confidence=0.5, min_lift=1.0):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_lift = min_lift
        self.frequent_itemsets = {}
        self.association_rules = []
        self.persona_rules = []
        self.approval_patterns = []
        
    def prepare_transaction_data(self, df):
        """Convert loan data into transaction format for association mining"""
        print("🔄 Preparing transaction data for association mining...")
        
        # Create bins for numerical features
        binning_rules = {}
        transactions = []
        
        for _, row in df.iterrows():
            transaction = []
            
            # Age categories
            age = row.get('PI_AGE', 30)
            if age < 25:
                transaction.append('Age_Young')
            elif age < 35:
                transaction.append('Age_Prime')
            elif age < 50:
                transaction.append('Age_Mature')
            else:
                transaction.append('Age_Senior')
            
            # Income categories
            income = row.get('PI_ANNUAL_INCOME', 500000)
            if income < 300000:
                transaction.append('Income_Low')
            elif income < 800000:
                transaction.append('Income_Medium')
            elif income < 1500000:
                transaction.append('Income_High')
            else:
                transaction.append('Income_VeryHigh')
            
            # Sum Assured categories
            sum_assured = row.get('SUM_ASSURED', 200000)
            if sum_assured < 100000:
                transaction.append('Sum_Small')
            elif sum_assured < 500000:
                transaction.append('Sum_Medium')
            elif sum_assured < 1000000:
                transaction.append('Sum_Large')
            else:
                transaction.append('Sum_VeryLarge')
            
            # Categorical features
            categorical_mappings = {
                'PI_GENDER': row.get('PI_GENDER', 'M'),
                'PI_OCCUPATION': row.get('PI_OCCUPATION', 'Salaried'),
                'ZONE': row.get('ZONE', 'Metro'),
                'PAYMENT_MODE': row.get('PAYMENT_MODE', 'Monthly'),
                'EARLY_NON': row.get('EARLY_NON', 'No'),
                'MEDICAL_NONMED': row.get('MEDICAL_NONMED', 'Medical'),
                'PI_STATE': row.get('PI_STATE', 'Delhi')
            }
            
            for key, value in categorical_mappings.items():
                if pd.notna(value):
                    transaction.append(f"{key}_{str(value).replace(' ', '_')}")
            
            # Approval status (if available)
            if 'POLICY_STATUS' in row:
                status = str(row['POLICY_STATUS']).lower()
                if 'approved' in status or 'claim' in status:
                    transaction.append('APPROVED')
                else:
                    transaction.append('REJECTED')
            
            # Income to Sum Assured ratio
            if income > 0 and sum_assured > 0:
                ratio = income / sum_assured
                if ratio > 10:
                    transaction.append('Ratio_Conservative')
                elif ratio > 5:
                    transaction.append('Ratio_Moderate')
                else:
                    transaction.append('Ratio_Aggressive')
            
            transactions.append(transaction)
        
        print(f"✅ Created {len(transactions)} transactions with avg {np.mean([len(t) for t in transactions]):.1f} items each")
        return transactions
    
    def find_frequent_itemsets(self, transactions):
        """Find frequent itemsets using Apriori algorithm"""
        print("🔍 Finding frequent itemsets...")
        
        # Get all unique items
        all_items = set()
        for transaction in transactions:
            all_items.update(transaction)
        
        total_transactions = len(transactions)
        min_support_count = int(self.min_support * total_transactions)
        
        # Level 1: Single items
        item_counts = {}
        for transaction in transactions:
            for item in transaction:
                item_counts[item] = item_counts.get(item, 0) + 1
        
        # Filter by minimum support
        frequent_1 = {frozenset([item]): count for item, count in item_counts.items() 
                     if count >= min_support_count}
        
        self.frequent_itemsets[1] = frequent_1
        print(f"   Level 1: {len(frequent_1)} frequent items")
        
        # Level 2 and beyond
        k = 2
        current_frequent = frequent_1
        
        while current_frequent and k <= 4:  # Limit to 4-itemsets for performance
            candidates = self._generate_candidates(list(current_frequent.keys()), k)
            candidate_counts = {}
            
            for transaction in transactions:
                transaction_set = set(transaction)
                for candidate in candidates:
                    if candidate.issubset(transaction_set):
                        candidate_counts[candidate] = candidate_counts.get(candidate, 0) + 1
            
            # Filter by minimum support
            current_frequent = {itemset: count for itemset, count in candidate_counts.items() 
                              if count >= min_support_count}
            
            if current_frequent:
                self.frequent_itemsets[k] = current_frequent
                print(f"   Level {k}: {len(current_frequent)} frequent itemsets")
            
            k += 1
        
        return self.frequent_itemsets
    
    def _generate_candidates(self, frequent_itemsets, k):
        """Generate candidate itemsets of size k"""
        candidates = []
        
        for i in range(len(frequent_itemsets)):
            for j in range(i + 1, len(frequent_itemsets)):
                # Join step
                set1 = frequent_itemsets[i]
                set2 = frequent_itemsets[j]
                
                # Union of the two sets
                candidate = set1.union(set2)
                
                if len(candidate) == k:
                    candidates.append(candidate)
        
        return list(set(candidates))
    
    def generate_association_rules(self, transactions):
        """Generate association rules from frequent itemsets"""
        print("📊 Generating association rules...")
        
        total_transactions = len(transactions)
        self.association_rules = []
        
        # For each frequent itemset of size >= 2
        for k in range(2, max(self.frequent_itemsets.keys()) + 1):
            if k not in self.frequent_itemsets:
                continue
                
            for itemset, support_count in self.frequent_itemsets[k].items():
                itemset_list = list(itemset)
                itemset_support = support_count / total_transactions
                
                # Generate all possible rules
                for i in range(1, len(itemset_list)):
                    for antecedent in combinations(itemset_list, i):
                        antecedent = frozenset(antecedent)
                        consequent = itemset - antecedent
                        
                        if len(antecedent) > 0 and len(consequent) > 0:
                            # Calculate confidence
                            antecedent_support = self._get_support(antecedent, total_transactions)
                            
                            if antecedent_support > 0:
                                confidence = itemset_support / antecedent_support
                                
                                # Calculate lift
                                consequent_support = self._get_support(consequent, total_transactions)
                                if consequent_support > 0:
                                    lift = confidence / consequent_support
                                    
                                    # Filter by thresholds
                                    if (confidence >= self.min_confidence and 
                                        lift >= self.min_lift and 
                                        itemset_support >= self.min_support):
                                        
                                        rule = {
                                            'antecedent': antecedent,
                                            'consequent': consequent,
                                            'support': itemset_support,
                                            'confidence': confidence,
                                            'lift': lift,
                                            'conviction': self._calculate_conviction(confidence, consequent_support)
                                        }
                                        self.association_rules.append(rule)
        
        # Sort by lift and confidence
        self.association_rules.sort(key=lambda x: (x['lift'], x['confidence']), reverse=True)
        print(f"✅ Generated {len(self.association_rules)} association rules")
        
        return self.association_rules
    
    def _get_support(self, itemset, total_transactions):
        """Get support for an itemset"""
        for k, frequent_sets in self.frequent_itemsets.items():
            if itemset in frequent_sets:
                return frequent_sets[itemset] / total_transactions
        return 0
    
    def _calculate_conviction(self, confidence, consequent_support):
        """Calculate conviction measure"""
        if confidence == 1.0:
            return float('inf')
        return (1 - consequent_support) / (1 - confidence) if confidence < 1 else 1
    
    def discover_personas(self):
        """Discover customer personas based on association rules"""
        print("👥 Discovering customer personas from association patterns...")
        
        # Group rules by patterns
        persona_patterns = {}
        approval_patterns = []
        
        for rule in self.association_rules:
            antecedent_str = '_'.join(sorted(rule['antecedent']))
            consequent_str = '_'.join(sorted(rule['consequent']))
            
            # Look for approval/rejection patterns
            if 'APPROVED' in rule['consequent'] or 'REJECTED' in rule['consequent']:
                approval_patterns.append(rule)
            
            # Group by similar antecedents for persona discovery
            if antecedent_str not in persona_patterns:
                persona_patterns[antecedent_str] = []
            persona_patterns[antecedent_str].append(rule)
        
        # Create persona definitions
        personas = []
        for i, (pattern, rules) in enumerate(persona_patterns.items()):
            if len(rules) >= 2:  # Only consider patterns with multiple rules
                persona = {
                    'id': i,
                    'name': self._generate_persona_name(rules[0]['antecedent']),
                    'pattern': rules[0]['antecedent'],
                    'characteristics': self._extract_characteristics(rules[0]['antecedent']),
                    'rules': rules,
                    'avg_confidence': np.mean([r['confidence'] for r in rules]),
                    'avg_lift': np.mean([r['lift'] for r in rules])
                }
                personas.append(persona)
        
        # Sort personas by strength (lift * confidence)
        personas.sort(key=lambda x: x['avg_lift'] * x['avg_confidence'], reverse=True)
        
        self.persona_rules = personas[:6]  # Keep top 6 personas
        self.approval_patterns = sorted(approval_patterns, key=lambda x: x['lift'], reverse=True)
        
        print(f"✅ Discovered {len(self.persona_rules)} customer personas")
        return self.persona_rules
    
    def _generate_persona_name(self, pattern):
        """Generate a descriptive name for a persona pattern"""
        pattern_list = list(pattern)
        
        # Extract key characteristics
        age_terms = [p for p in pattern_list if p.startswith('Age_')]
        income_terms = [p for p in pattern_list if p.startswith('Income_')]
        occupation_terms = [p for p in pattern_list if p.startswith('PI_OCCUPATION_')]
        zone_terms = [p for p in pattern_list if p.startswith('ZONE_')]
        
        name_parts = []
        
        if age_terms:
            age = age_terms[0].replace('Age_', '')
            name_parts.append(age)
        
        if income_terms:
            income = income_terms[0].replace('Income_', '')
            name_parts.append(income + ' Income')
        
        if occupation_terms:
            occupation = occupation_terms[0].replace('PI_OCCUPATION_', '').replace('_', ' ')
            name_parts.append(occupation)
        
        if zone_terms:
            zone = zone_terms[0].replace('ZONE_', '')
            name_parts.append(zone + ' Zone')
        
        if name_parts:
            return ' '.join(name_parts[:2])  # Limit to 2 main characteristics
        else:
            return f"Customer Segment"
    
    def _extract_characteristics(self, pattern):
        """Extract human-readable characteristics from a pattern"""
        characteristics = {}
        
        for item in pattern:
            if item.startswith('Age_'):
                characteristics['age_group'] = item.replace('Age_', '').lower()
            elif item.startswith('Income_'):
                characteristics['income_level'] = item.replace('Income_', '').lower()
            elif item.startswith('PI_OCCUPATION_'):
                characteristics['occupation'] = item.replace('PI_OCCUPATION_', '').replace('_', ' ')
            elif item.startswith('ZONE_'):
                characteristics['zone'] = item.replace('ZONE_', '')
            elif item.startswith('Ratio_'):
                characteristics['risk_profile'] = item.replace('Ratio_', '').lower()
        
        return characteristics
    
    def predict_persona(self, customer_data):
        """Predict customer persona based on association rules"""
        # Convert customer data to transaction format
        transaction = self._customer_to_transaction(customer_data)
        transaction_set = set(transaction)
        
        # Find matching personas
        persona_scores = []
        
        for persona in self.persona_rules:
            pattern = persona['pattern']
            overlap = len(pattern.intersection(transaction_set))
            coverage = overlap / len(pattern) if len(pattern) > 0 else 0
            
            if coverage > 0.5:  # At least 50% pattern match
                score = coverage * persona['avg_confidence'] * persona['avg_lift']
                persona_scores.append((persona['id'], score, persona))
        
        if persona_scores:
            # Return best matching persona
            persona_scores.sort(key=lambda x: x[1], reverse=True)
            return persona_scores[0][2]
        else:
            # Return default persona if no match
            return {
                'id': 0,
                'name': 'Standard Customer',
                'characteristics': {'profile': 'standard'},
                'avg_confidence': 0.5,
                'avg_lift': 1.0
            }
    
    def _customer_to_transaction(self, customer_data):
        """Convert customer data to transaction format"""
        transaction = []
        
        # Age
        age = customer_data.get('PI_AGE', 30)
        if age < 25:
            transaction.append('Age_Young')
        elif age < 35:
            transaction.append('Age_Prime')
        elif age < 50:
            transaction.append('Age_Mature')
        else:
            transaction.append('Age_Senior')
        
        # Income
        income = customer_data.get('PI_ANNUAL_INCOME', 500000)
        if income < 300000:
            transaction.append('Income_Low')
        elif income < 800000:
            transaction.append('Income_Medium')
        elif income < 1500000:
            transaction.append('Income_High')
        else:
            transaction.append('Income_VeryHigh')
        
        # Sum Assured
        sum_assured = customer_data.get('SUM_ASSURED', 200000)
        if sum_assured < 100000:
            transaction.append('Sum_Small')
        elif sum_assured < 500000:
            transaction.append('Sum_Medium')
        elif sum_assured < 1000000:
            transaction.append('Sum_Large')
        else:
            transaction.append('Sum_VeryLarge')
        
        # Categorical features
        categorical_mappings = {
            'PI_GENDER': customer_data.get('PI_GENDER', 'M'),
            'PI_OCCUPATION': customer_data.get('PI_OCCUPATION', 'Salaried'),
            'ZONE': customer_data.get('ZONE', 'Metro'),
            'PAYMENT_MODE': customer_data.get('PAYMENT_MODE', 'Monthly'),
            'EARLY_NON': customer_data.get('EARLY_NON', 'No'),
            'MEDICAL_NONMED': customer_data.get('MEDICAL_NONMED', 'Medical'),
            'PI_STATE': customer_data.get('PI_STATE', 'Delhi')
        }
        
        for key, value in categorical_mappings.items():
            if pd.notna(value):
                transaction.append(f"{key}_{str(value).replace(' ', '_')}")
        
        # Ratio
        if income > 0 and sum_assured > 0:
            ratio = income / sum_assured
            if ratio > 10:
                transaction.append('Ratio_Conservative')
            elif ratio > 5:
                transaction.append('Ratio_Moderate')
            else:
                transaction.append('Ratio_Aggressive')
        
        return transaction
    
    def get_approval_insights(self):
        """Get insights about approval patterns"""
        insights = []
        
        for rule in self.approval_patterns[:10]:  # Top 10 patterns
            if 'APPROVED' in rule['consequent']:
                insight = {
                    'type': 'approval',
                    'pattern': list(rule['antecedent']),
                    'confidence': rule['confidence'],
                    'lift': rule['lift'],
                    'description': f"Customers with {', '.join(rule['antecedent'])} have {rule['confidence']*100:.1f}% approval rate"
                }
                insights.append(insight)
        
        return insights
    
    def generate_recommendations(self, customer_data):
        """Generate recommendations based on association rules"""
        persona = self.predict_persona(customer_data)
        recommendations = []
        
        # Find applicable approval patterns
        customer_transaction = set(self._customer_to_transaction(customer_data))
        
        for rule in self.approval_patterns:
            if 'APPROVED' in rule['consequent']:
                antecedent = rule['antecedent']
                missing_items = antecedent - customer_transaction
                
                if len(missing_items) <= 2 and len(missing_items) > 0:  # 1-2 missing items
                    recommendation = {
                        'suggestion': f"Consider: {', '.join(missing_items)}",
                        'reason': f"This pattern has {rule['confidence']*100:.1f}% approval rate",
                        'confidence': rule['confidence'],
                        'lift': rule['lift']
                    }
                    recommendations.append(recommendation)
        
        # Sort by confidence
        recommendations.sort(key=lambda x: x['confidence'], reverse=True)
        
        return {
            'persona': persona,
            'recommendations': recommendations[:5]  # Top 5 recommendations
        } 