f_bu_eb_1 = open('bu_eb_1.csv', 'w')
f_bu_eb_2 = open('bu_eb_2.csv', 'w')

for i in xrange(1, 17):
    f_bu_eb_1.write('BU/L%i/s-EB/(LR[1-8],[M,A],1)/b\n' % i)
    f_bu_eb_2.write('bu/R%i/s-EB/(LR[1-8],[M,A],1)/b\n' % i)
for i in xrange(33, 49):
    f_bu_eb_1.write('BU/L%i/s-EB/(LR[1-8],A,[3,4])/b\n' % i)
    f_bu_eb_2.write('bu/R%i/s-EB/(LR[1-8],A,[3,4])/b\n' % i)
for i in xrange(17, 33):
    f_bu_eb_1.write('BU/L%i/s-EB/(LR[1-8],A,[1,2])/b\n' % i)
    f_bu_eb_2.write('bu/R%i/s-EB/(LR[1-8],A,[1,2])/b\n' % i)
for i in xrange(49, 65):
    f_bu_eb_1.write('BU/L%i/s-EB/(LR[1-8],A,4)/b\n' % i)
    f_bu_eb_2.write('bu/R%i/s-EB/(LR[1-8],A,4)/b\n' % i)
for i in xrange(65, 81):
    f_bu_eb_1.write('BU/L%i/s-EB/(LR[1-8],M,4)/b\n' % i)
    f_bu_eb_2.write('bu/R%i/s-EB/(LR[1-8],M,4)/b\n' % i)
f_bu_eb_1.close()
f_bu_eb_2.close()
