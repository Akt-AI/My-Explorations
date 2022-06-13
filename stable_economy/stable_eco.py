import json

#People may generate more than one identity.
def generate_identity():
	pass
	#return private and public key. Make a identity.

def create_your_profile():
	pass
	#optional
		
class Universal_Registry():
	pass

class Local_groups():
	pass

class StackHolder_communications():
	pass
	
class Public_Apis():
	pass
	
class StackHolder:
	def __init__(self):
		print("Enter your Public Key:")
		self.pub_key = input()
		print("Enter Name of StackHolder:")
		self.name = input()
		self.fund = 0
		self.fund += 3000 
		print("Enter StackHolder type, e.g Public, Private, Individual")
		self.type = input()
		print("Your identity is registered.")
		
		
#Equal Amount of fund will be released by 
#Central Bank when an individual will joint the network to Public fund.
#if identity is generated: Release a fund to account
#Its not anonymous. 
	
minimal_stackholder_fund = 3000
public_fund = 0
social_responsibility = 0
universal_basic_income = 0
total_fund = 0

def public_welfare_fund():
	pass
	
def strip_individual_fund_if_crossed_the_limit():
	pass
	#Add the fund to public welfare fund
	
def wallet(identity):
	if stackholder.fund > minimal_stackholder_fund:
	diff = minimal_stackholder_fund - stackholder.fund 
	public_fund = public_fund + diff
	#Each stackholder will have a wallet.
	pass

def monitor_stackholder_fund():
	if stackholder.fund < minimum_label_of_stackholder_fund:
		diff = public_fund - social_responsibility
		stackholder.fund = stackholder.fund + diff
			
def universal_publicly_verifiable_ledger():
	pass
	
def institutions(identity):
	pass
	
def request_for_release_of_fund():
	pass

def approval_of_fund():
	pass
	
def accessment_of_total_need():
	pass
	
def release_fund_to_institution():
	#create Institution or Manufacturing Unit
	#Pay Salary
	#Ensure Minimum Cost of living.
	#Maximum Wage Differance in mental and physical labour.
	#Institutions
	#Judicial Cost
	#Ensure Maximum democratic participation.
	#Release fund by voting.
	pass
	
def prooof_of_work_in_real_sence():
	pass
	
def fix_your_price():
	pass
	
def proof_of_delvery_of_service():
	#Fix the price of your service
	#Fix Min-Max price of Service
	pass

def proof_of_delivery_of_goods():
	#Fix the price of your goods
	#Fix Min-Max price of goods
	pass
	
def public_standard_price_list():
	pass
	
def show_fund():
	pass
	#This will show the each stack-holders funds.
	
def allow_competition():
	#return Bool
	pass
	
def automatic_release_of_fund_in_crisis():
	pass
	#Show the evidence of crisis.
	
def orphan_funds():
	pass  # Persons left or died or not active in economy Trip the orphan fund.
	#Strip orphan funds after one year.
	
def binary_transactions():
	pub_fund=[]
	credit = (stackholder.pub_key, stackholder.pub_key, transaction_fee)
	debit = (stackholder.pub_key, stackholder.pub_key, transaction_fee)
	pub_fund.append(transaction_fee) 
	pass
	
def ensure_security_of_registration_and_transaction_portal():
	pass
	
while True:	
	stackholder = StackHolder()
	entry = {}
	entry['pubkey'] = stackholder.pub_key
	entry['fund'] = stackholder.fund
	entry['Type'] = stackholder.type
	total_fund +=3000
	entry['total_fund_in_economy'] = 2*total_fund
	print("total_fund: ", 2*total_fund)
	with open('register.json', 'a') as f:
		f.write(json.dumps(entry, indent=4))
		 
